#include <depthcompletion/depthcompletion.hpp>
#include <realsense_camera/camera.hpp>

#include <torch/torch.h>

#include <Eigen/Geometry>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace depthcompletion {

class DepthCompletionNodelet : public rclcpp::Node {
public:
    explicit DepthCompletionNodelet(const rclcpp::NodeOptions &options)
        : Node("depthcompletion", options), _clock(RCL_ROS_TIME) {

        declareAndGetParameters();
        initializeEngine();
        initializeCamera();
        initializePublishers();
        startFrameTimer();
    }

private:
    template <typename T>
    void loadParameter(const std::string &param_name, T &param_value, const std::string &format) {
        if (!this->get_parameter(param_name, param_value))
            RCLCPP_ERROR(this->get_logger(), "[DepthCompletion] No %s!", param_name.c_str());
        else {
            if constexpr (std::is_same_v<T, std::string>)
                RCLCPP_INFO(this->get_logger(), "[DepthCompletion] %s: %s", param_name.c_str(), param_value.c_str());
            else
                RCLCPP_INFO(this->get_logger(), ("[DepthCompletion] " + param_name + ": " + format).c_str(), param_value);
        }
    }
    
    void declareAndGetParameters() {
        this->declare_parameter("workspace_path", "");
        this->declare_parameter("engine.relative_path", "");
        loadParameter("workspace_path", _ws_path, "%s");
        loadParameter("engine.relative_path", _engine_path, "%s");
        _engine_path = _ws_path + _engine_path;

        this->declare_parameter("engine.width", 0);
        this->declare_parameter("engine.height", 0);
        this->declare_parameter("engine.batchsize", 0);
        loadParameter("engine.width", _engine_width, "%d");
        loadParameter("engine.height", _engine_height, "%d");
        loadParameter("engine.batchsize", _engine_batchsize, "%d");

        this->declare_parameter("camera.frame_id", "");
        this->declare_parameter("camera.width", 0);
        this->declare_parameter("camera.height", 0);
        this->declare_parameter("camera.channels", 0);
        this->declare_parameter("camera.fps", 0);
        this->declare_parameter("camera.min_range", 0.0);
        this->declare_parameter("camera.max_range", 0.0);
        this->declare_parameter("camera.speckle_max_size", 0);
        this->declare_parameter("camera.speckle_diff", 0);
        loadParameter("camera.frame_id", _camera_frame_id, "%s");
        loadParameter("camera.width",  _camera_width, "%d");
        loadParameter("camera.height", _camera_height, "%d");
        loadParameter("camera.channels", _camera_channels, "%d");
        loadParameter("camera.fps", _camera_fps, "%d");
        loadParameter("camera.min_range", _camera_min_range, "%.4f");
        loadParameter("camera.max_range", _camera_max_range, "%.4f");
        loadParameter("camera.speckle_max_size", _speckle_max_size, "%d");
        loadParameter("camera.speckle_diff", _speckle_diff, "%d");
    }

    void initializeEngine() {
        _mde_engine.init(_engine_path, _engine_batchsize, _engine_height, _engine_width, _camera_channels);
    }

    void initializeCamera() {
        _camera.init(_ws_path, _camera_width, _camera_height, _camera_fps);

        float camera_width, camera_height, camera_norm_fx, camera_norm_fy, camera_norm_cx, camera_norm_cy;
        _camera.getNormalizedParameters(camera_width, camera_height, camera_norm_fx, camera_norm_fy, camera_norm_cx, camera_norm_cy);
        _camera_fx = camera_norm_fx * _camera_width;
        _camera_fy = camera_norm_fy * _camera_height;
        _camera_cx = camera_norm_cx * _camera_width;
        _camera_cy = camera_norm_cy * _camera_height;
    }

    void initializePublishers() {
        _pub_color = this->create_publisher<sensor_msgs::msg::Image>("rs_camera/color/image_raw", 1);
        _pub_depth = this->create_publisher<sensor_msgs::msg::Image>("rs_camera/depth/image_raw", 1);
    }
    
    void startFrameTimer() {
        _timer = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / _camera_fps)),
            std::bind(&DepthCompletionNodelet::grabAndCompleteDepth, this));
    }

    void grabAndCompleteDepth() {

        // Load frames
        uint64_t timestamp;
        cv::Mat color_image, depth_image;
        if (!_camera.grabFrames(color_image, depth_image, timestamp)) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), _clock, 1000,
                "[DepthCompletion] Dropped frame longing than RealSense API timeout!");
            return;
        }

        // Resize frames
        cv::resize(color_image, color_image, cv::Size(_engine_width, _engine_height), 0, 0, cv::INTER_LINEAR);
        cv::resize(depth_image, depth_image, cv::Size(_engine_width, _engine_height), 0, 0, cv::INTER_LINEAR);

        // Change unit: mm to m
        depth_image.convertTo(depth_image, CV_32F, 1.0 / 1000.0);

        // Filter speckle noise
        cv::Mat depth_image_speckles;
        depth_image.convertTo(depth_image_speckles, CV_16SC1);
        cv::filterSpeckles(depth_image_speckles, -1, _speckle_max_size, _speckle_diff);

        // Sensor disparity
        cv::Mat disparity_image;
        cv::divide(1.0f, depth_image + 1e-3f, disparity_image);

        // Monocular disparity estimation
        cv::Mat pred_disparity = _mde_engine.predict(color_image);

        // Optimize the two clouds together to find the alignment factor
        cv::Mat mask = (pred_disparity > 0.0f) & (disparity_image < 1000.0f) & (depth_image > _camera_min_range) & (depth_image < _camera_max_range) & (depth_image_speckles != -1);
        cv::Mat masked_pred_disparity, masked_disparity_image;
        pred_disparity.copyTo(masked_pred_disparity, mask);
        disparity_image.copyTo(masked_disparity_image, mask);

        cv::Mat masked_pred_disparity_flattened, masked_disparity_image_flattened;
        masked_pred_disparity_flattened = masked_pred_disparity.reshape(1, pred_disparity.total());
        masked_disparity_image_flattened = masked_disparity_image.reshape(1, pred_disparity.total());

        cv::Mat masked_pred_disparity_sq_flattened;
        cv::multiply(masked_pred_disparity_flattened, masked_pred_disparity_flattened, masked_pred_disparity_sq_flattened);

        cv::Mat X = cv::Mat::ones(pred_disparity.total(), 3, CV_32F);
        masked_pred_disparity_flattened.copyTo(X.col(1));
        masked_pred_disparity_sq_flattened.copyTo(X.col(2));
        cv::Mat XtX = X.t() * X;
        cv::Mat XtX_inv;
        cv::invert(XtX, XtX_inv, cv::DECOMP_SVD);
        cv::Mat factor = XtX_inv * X.t() * masked_disparity_image_flattened;

        cv::Mat pred_disparity_sq;
        cv::multiply(pred_disparity, pred_disparity, pred_disparity_sq);
        cv::Mat completed_disparity = pred_disparity_sq * factor.at<float>(2, 0) + pred_disparity * factor.at<float>(1, 0) + factor.at<float>(0, 0);

        // Disparity to depth
        cv::Mat completed_depth;
        cv::divide(1.0f, completed_disparity + 1e-3f, completed_depth);

        // Publish the depth and color frames
        cv::Mat normalized_depth;
        cv::normalize(completed_depth, normalized_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        sensor_msgs::msg::Image::SharedPtr color_msg = 
            cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::RGB8, color_image).toImageMsg();
        sensor_msgs::msg::Image::SharedPtr depth_msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::TYPE_8UC1, normalized_depth).toImageMsg();
        rclcpp::Time t = _clock.now();
        color_msg->header.stamp = t;
        depth_msg->header.stamp = t;
        color_msg->header.frame_id = _camera_frame_id;
        depth_msg->header.frame_id = _camera_frame_id;
        _pub_color->publish(*color_msg);
        _pub_depth->publish(*depth_msg);
    }

    // Node parameters
    std::string _ws_path;
    std::string _engine_path;
    int _engine_width;
    int _engine_height;
    int _engine_batchsize;
    std::string _camera_frame_id;
    int _camera_width;
    int _camera_height;
    int _camera_channels;
    int _camera_fps;
    float _camera_min_range;
    float _camera_max_range;
    int _speckle_max_size;
    int _speckle_diff;

    // Engine instance
    DepthCompletionEngine _mde_engine;

    // Camera instance
    realsense_camera::RSCamera _camera;
    float _camera_fx;
    float _camera_fy;
    float _camera_cx;
    float _camera_cy;

    // ROS communication
    rclcpp::Clock _clock;
    rclcpp::TimerBase::SharedPtr _timer;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_color;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_depth;
};

}  // namespace depthcompletion

RCLCPP_COMPONENTS_REGISTER_NODE(depthcompletion::DepthCompletionNodelet)