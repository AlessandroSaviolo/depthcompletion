#include <realsense_camera/camera.hpp>
#include <depthcompletion/depthcompletion.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace depthcompletion {

class DepthCompletionNodelet : public rclcpp::Node {
public:
    explicit DepthCompletionNodelet(const rclcpp::NodeOptions &options)
        : Node("depth_completion", options), 
          _clock(RCL_ROS_TIME), 
          _epsilon(1e-3f), 
          _running_factor_alpha(0.1f),
          _running_factor_mean(cv::Mat::zeros(3, 1, CV_32F)) {

        // Initialize all components
        declareParameters();
        validateAndLoadParameters();
        initializeEngine();
        initializeCamera();
        setupPublishers();
        startTimer();
    }

private:
    void declareParameters() {
        // Declare parameters with default values
        this->declare_parameter<std::string>("workspace_path", "");
        this->declare_parameter<std::string>("engine.relative_path", "");
        this->declare_parameter<int>("engine.width", 0);
        this->declare_parameter<int>("engine.height", 0);
        this->declare_parameter<int>("engine.batchsize", 0);
        this->declare_parameter<std::string>("camera.frame_id", "");
        this->declare_parameter<int>("camera.width", 0);
        this->declare_parameter<int>("camera.height", 0);
        this->declare_parameter<int>("camera.channels", 0);
        this->declare_parameter<int>("camera.fps", 0);
        this->declare_parameter<double>("camera.min_range", 0.0f);
        this->declare_parameter<double>("camera.max_range", 0.0f);
        this->declare_parameter<int>("camera.speckle_max_size", 0);
        this->declare_parameter<int>("camera.speckle_diff", 0);
    }

    void validateAndLoadParameters() {
        // Load and validate workspace and engine parameters
        if (!get_parameter("workspace_path", _ws_path) || _ws_path.empty()) {
            RCLCPP_FATAL(this->get_logger(), "Parameter 'workspace_path' is not set or empty.");
            throw std::runtime_error("Missing required parameter: workspace_path");
        }

        std::string engine_relative_path;
        if (!get_parameter("engine.relative_path", engine_relative_path) || engine_relative_path.empty()) {
            RCLCPP_FATAL(this->get_logger(), "Parameter 'engine.relative_path' is not set or empty.");
            throw std::runtime_error("Missing required parameter: engine.relative_path");
        }
        _engine_path = _ws_path + engine_relative_path;

        // Load engine dimensions and batch size
        loadValidatedParameter("engine.width", _engine_width, 1);
        loadValidatedParameter("engine.height", _engine_height, 1);
        loadValidatedParameter("engine.batchsize", _engine_batchsize, 1);

        // Load camera parameters
        loadValidatedParameter("camera.frame_id", _camera_frame_id);
        loadValidatedParameter("camera.width", _camera_width, 1);
        loadValidatedParameter("camera.height", _camera_height, 1);
        loadValidatedParameter("camera.channels", _camera_channels, 1);
        loadValidatedParameter("camera.fps", _camera_fps, 1);
        loadValidatedParameter("camera.min_range", _camera_min_range, 0.0f);
        loadValidatedParameter("camera.max_range", _camera_max_range, 0.0f);
        loadValidatedParameter("camera.speckle_max_size", _speckle_max_size, 0);
        loadValidatedParameter("camera.speckle_diff", _speckle_diff, 0);
    }

    template <typename T>
    void loadValidatedParameter(const std::string &param_name, T &param_value, T min_value = T()) {
        if (!get_parameter(param_name, param_value) || param_value < min_value) {
            RCLCPP_FATAL(this->get_logger(), "Invalid or missing parameter: %s", param_name.c_str());
            throw std::runtime_error("Invalid or missing parameter: " + param_name);
        }

        // Check type at compile-time and handle accordingly
        if constexpr (std::is_same_v<T, std::string>) {
            RCLCPP_INFO(this->get_logger(), "%s: %s", param_name.c_str(), param_value.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "%s: %s", param_name.c_str(), std::to_string(param_value).c_str());
        }
    }

    void initializeEngine() {
        try {
            _mde_engine.init(_engine_path, _engine_batchsize, _engine_height, _engine_width, _camera_channels);
            RCLCPP_INFO(this->get_logger(), "DepthCompletionEngine initialized successfully.");
        } catch (const std::exception &e) {
            RCLCPP_FATAL(this->get_logger(), "Engine initialization failed: %s", e.what());
            throw;
        }
    }

    void initializeCamera() {
        try {
            _camera.init(_ws_path, _camera_width, _camera_height, _camera_fps);
            
            // Fetch camera parameters
            float camera_norm_width, camera_norm_height, camera_norm_fx, camera_norm_fy, camera_norm_cx, camera_norm_cy;
            _camera.getNormalizedParameters(
                camera_norm_width, camera_norm_height, camera_norm_fx, camera_norm_fy, camera_norm_cx, camera_norm_cy);

            _camera_fx = camera_norm_fx * _camera_width;
            _camera_fy = camera_norm_fy * _camera_height;
            _camera_cx = camera_norm_cx * _camera_width;
            _camera_cy = camera_norm_cy * _camera_height;

            RCLCPP_INFO(this->get_logger(), "Camera initialized with focal lengths: fx=%.2f, fy=%.2f", _camera_fx, _camera_fy);
        } 
        catch (const std::exception &e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to initialize RealSense camera: %s", e.what());
            throw;
        }
    }

    void setupPublishers() {
        _pub_color = this->create_publisher<sensor_msgs::msg::Image>("camera/color/image_raw", 10);
        _pub_sparse_depth = this->create_publisher<sensor_msgs::msg::Image>("camera/depth/image_raw", 10);
        _pub_relative_depth = this->create_publisher<sensor_msgs::msg::Image>("camera/depth/relative", 10);
        _pub_completed_depth = this->create_publisher<sensor_msgs::msg::Image>("camera/depth/completed", 10);
        _pub_sparse_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("camera/pointcloud/image_raw", 10);
        _pub_relative_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("camera/pointcloud/relative", 10);
        _pub_completed_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("camera/pointcloud/completed", 10);
    }

    void startTimer() {
        _timer = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / _camera_fps)),
            std::bind(&DepthCompletionNodelet::processFrames, this));
    }

    void processFrames() {
        uint64_t timestamp;
        cv::Mat color_image, depth_image;

        // Capture frames
        if (!_camera.grabFrames(color_image, depth_image, timestamp)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), _clock, 1000,
                "Dropped frame due to camera capture timeout.");
            return;
        }

        // Resize images for processing
        cv::resize(color_image, color_image, cv::Size(_engine_width, _engine_height), 0, 0, cv::INTER_LINEAR);
        cv::resize(depth_image, depth_image, cv::Size(_engine_width, _engine_height), 0, 0, cv::INTER_LINEAR);

        // Convert depth units
        depth_image.convertTo(depth_image, CV_32F, 1.0 / 1000.0);

        // Perform depth completion
        cv::Mat relative_depth, completed_depth;
        completeDepth(color_image, depth_image, relative_depth, completed_depth);

        // Publish results
        publishFrames(color_image, depth_image, relative_depth, completed_depth);
    }

    void completeDepth(cv::Mat color_image, cv::Mat depth_image,
                       cv::Mat& relative_depth, cv::Mat& completed_depth) {

        // Sensor disparity
        cv::Mat disparity_image;
        cv::divide(1.0f, depth_image + _epsilon, disparity_image);

        // Filter speckle noise
        cv::Mat depth_image_speckles;
        depth_image.convertTo(depth_image_speckles, CV_16SC1);
        cv::filterSpeckles(depth_image_speckles, -1, _speckle_max_size, _speckle_diff);

        // Monocular disparity estimation
        cv::Mat pred_disparity = _mde_engine.predict(color_image);

        // Optimize the two clouds together to find the alignment factor
        cv::Mat mask = (depth_image > _camera_min_range) & 
                       (depth_image < _camera_max_range) & 
                       (depth_image_speckles != -1);
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

        // If the factor is not all zeros, update the running mean
        if (!cv::countNonZero(factor) == 0) {
            _running_factor_mean = _running_factor_mean * (1.0f - _running_factor_alpha) + factor * _running_factor_alpha;
        }

        cv::Mat pred_disparity_sq;
        cv::multiply(pred_disparity, pred_disparity, pred_disparity_sq);
        cv::Mat completed_disparity = _running_factor_mean.at<float>(2, 0) * pred_disparity_sq + 
                                      _running_factor_mean.at<float>(1, 0) * pred_disparity + 
                                      _running_factor_mean.at<float>(0, 0);

        // Disparity to depth
        cv::divide(1.0f, pred_disparity + _epsilon, relative_depth);
        cv::divide(1.0f, completed_disparity + _epsilon, completed_depth);

        return;
    }

    void publishFrames(cv::Mat &color_image, cv::Mat &sparse_depth, 
                       cv::Mat &relative_depth, cv::Mat &completed_depth) {

        auto timestamp = _clock.now();

        auto color_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", color_image).toImageMsg();
        color_msg->header.stamp = timestamp;
        color_msg->header.frame_id = _camera_frame_id;

        auto sparse_depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", sparse_depth).toImageMsg();
        sparse_depth_msg->header.stamp = timestamp;
        sparse_depth_msg->header.frame_id = _camera_frame_id;
     
        auto relative_depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", relative_depth).toImageMsg();
        relative_depth_msg->header.stamp = timestamp;
        relative_depth_msg->header.frame_id = _camera_frame_id;
     
        auto completed_depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", completed_depth).toImageMsg();
        completed_depth_msg->header.stamp = timestamp;
        completed_depth_msg->header.frame_id = _camera_frame_id;

        _pub_color->publish(*color_msg);
        _pub_sparse_depth->publish(*sparse_depth_msg);
        _pub_relative_depth->publish(*relative_depth_msg);
        _pub_completed_depth->publish(*completed_depth_msg);

        Eigen::MatrixXf sparse_cloud = depthToPointCloud(sparse_depth);
        sensor_msgs::msg::PointCloud2 sparse_cloud_msg = getCloudMsg(sparse_cloud);
        _pub_sparse_cloud->publish(sparse_cloud_msg);

        Eigen::MatrixXf relative_cloud = depthToPointCloud(relative_depth);
        sensor_msgs::msg::PointCloud2 relative_cloud_msg = getCloudMsg(relative_cloud);
        _pub_relative_cloud->publish(relative_cloud_msg);

        Eigen::MatrixXf completed_cloud = depthToPointCloud(completed_depth);
        sensor_msgs::msg::PointCloud2 completed_cloud_msg = getCloudMsg(completed_cloud);
        _pub_completed_cloud->publish(completed_cloud_msg);
    }

    Eigen::MatrixXf depthToPointCloud(const cv::Mat& depth) {
        std::vector<Eigen::Vector3f> points;
        for (int v = 0; v < depth.rows; ++v) {
            for (int u = 0; u < depth.cols; ++u) {
                float z = depth.at<float>(v, u);
                if (std::isfinite(z) && z > 0.1f) {
                    float x = (u - _camera_cx) * z / _camera_fx;
                    float y = (v - _camera_cy) * z / _camera_fy;
                    points.emplace_back(x, y, z);
                }
            }
        }

        Eigen::MatrixXf cloud(3, points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            cloud.col(i) = points[i];
        }
        return cloud;
    }

    sensor_msgs::msg::PointCloud2 getCloudMsg(
        const Eigen::MatrixXf& point_cloud) {
            
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = _clock.now();
        cloud_msg.header.frame_id = "cam1_link";
        cloud_msg.width = point_cloud.cols();
        cloud_msg.height = 1;
        cloud_msg.is_dense = true;
        cloud_msg.is_bigendian = false;
        cloud_msg.fields.resize(3);
        cloud_msg.fields[0].name = "x";
        cloud_msg.fields[1].name = "y";
        cloud_msg.fields[2].name = "z";

        int offset = 0;
        for (size_t i = 0; i < cloud_msg.fields.size(); ++i, offset += 4) {
            cloud_msg.fields[i].offset = offset;
            cloud_msg.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
            cloud_msg.fields[i].count = 1;
        }
        cloud_msg.point_step = offset; // Length of a point in bytes

        // Calculate the row_step
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

        // Resize data to fit all points
        cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

        // Fill in the point cloud data
        float* data_ptr = reinterpret_cast<float*>(cloud_msg.data.data());
        for (int i = 0; i < point_cloud.cols(); ++i) {
            data_ptr[3 * i]     = point_cloud(0, i);
            data_ptr[3 * i + 1] = point_cloud(1, i);
            data_ptr[3 * i + 2] = point_cloud(2, i);
        }
        return cloud_msg;
    }

    // Small constant
    float _epsilon;
    
    // Running mean output from MDE optimizatio
    cv::Mat _running_factor_mean;
    float _running_factor_alpha;

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

    // Engine and camera instances
    DepthCompletionEngine _mde_engine;
    realsense_camera::RSCamera _camera;

    // Camera intrinsic parameters
    float _camera_fx, _camera_fy, _camera_cx, _camera_cy;

    // ROS communication
    rclcpp::Clock _clock;
    rclcpp::TimerBase::SharedPtr _timer;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_color;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_sparse_depth;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_relative_depth;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_completed_depth;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_sparse_cloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_relative_cloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_completed_cloud;
};

}  // namespace depthcompletion

RCLCPP_COMPONENTS_REGISTER_NODE(depthcompletion::DepthCompletionNodelet)
