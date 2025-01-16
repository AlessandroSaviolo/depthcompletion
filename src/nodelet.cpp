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

namespace arpl_depth_completion {

class DepthCompletionNodelet : public rclcpp::Node {
public:
    DepthCompletionNodelet(const rclcpp::NodeOptions &options)
        : Node("arpl_depth_completion", options),
          _is_odom_init(false),
          _epsilon(1e-3) { onInit(); }

    void onInit() {
        this->declare_parameter("workspace_path", "");
        loadParameter("workspace_path", _ws_path, "%s");

        this->declare_parameter("control_odom", "");
        loadParameter("control_odom", _control_odom_topic, "%s");

        this->declare_parameter("world_frame_id", "");
        this->declare_parameter("camera_frame_id", "");
        loadParameter("world_frame_id", _world_frame_id, "%s");
        loadParameter("camera_frame_id", _camera_frame_id, "%s");

        this->declare_parameter("nmpc.nc", 0);
        loadParameter("nmpc.nc", _num_avoid_points, "%d");
        _num_avoid_points = _num_avoid_points / 3;

        this->declare_parameter("pooling.kernel.width", 0);
        this->declare_parameter("pooling.kernel.height", 0);
        loadParameter("pooling.kernel.width", _pooling_width, "%d");
        loadParameter("pooling.kernel.height", _pooling_height, "%d");

        this->declare_parameter("engine.mde.width", 0);
        this->declare_parameter("engine.mde.height", 0);
        this->declare_parameter("engine.batch_size", 0);
        loadParameter("engine.mde.width", _engine_width, "%d");
        loadParameter("engine.mde.height", _engine_height, "%d");
        loadParameter("engine.batch_size", _batch_size, "%d");

        this->declare_parameter("engine.mde.relative_path", "");
        loadParameter("engine.mde.relative_path", _engine_path, "%s");
        _engine_path = _ws_path + _engine_path;

        this->declare_parameter("camera.width", 0);
        this->declare_parameter("camera.height", 0);
        this->declare_parameter("camera.channels", 0);
        this->declare_parameter("camera.min_range", 0.0);
        this->declare_parameter("camera.max_range", 0.0);
        this->declare_parameter("camera.speckle_max_size", 0);
        this->declare_parameter("camera.speckle_diff", 0);
        loadParameter("camera.width",  _camera_width, "%d");
        loadParameter("camera.height", _camera_height, "%d");
        loadParameter("camera.channels", _camera_channels, "%d");
        loadParameter("camera.min_range", _camera_min_range, "%.4f");
        loadParameter("camera.max_range", _camera_max_range, "%.4f");
        loadParameter("camera.speckle_max_size", _speckle_max_size, "%d");
        loadParameter("camera.speckle_diff", _speckle_diff, "%d");

        this->declare_parameter("robot.width", 0.0);
        this->declare_parameter("robot.height", 0.0);
        loadParameter("robot.width",  _robot_width, "%.4f");
        loadParameter("robot.height", _robot_height, "%.4f");

        this->declare_parameter("nmpc.max_obs_rel_pos", 0.0);
        this->declare_parameter("nmpc.max_obs_rel_vel", 0.0);
        this->declare_parameter("nmpc.max_obs_rel_acc", 0.0);
        loadParameter("nmpc.max_obs_rel_pos", _max_obs_rel_pos, "%f");
        loadParameter("nmpc.max_obs_rel_vel", _max_obs_rel_vel, "%f");
        loadParameter("nmpc.max_obs_rel_acc", _max_obs_rel_acc, "%f");

        _t_W_B = Eigen::Vector3f::Zero();
        _v_W = Eigen::Vector3f::Zero();
        _R_W_B = Eigen::Matrix3f::Identity();
        _q_B_C = Eigen::Quaternionf(-0.5, 0.5, -0.5, 0.5);
        _q_C_B = _q_B_C.conjugate();
        _R_B_C = _q_B_C.toRotationMatrix();
        _t_B_C = Eigen::Vector3f(0.5, 0.3, -0.3);
        _t_W_B_z_0 = 0.0;

        _mde_engine.init(_engine_path, _batch_size, _engine_height, _engine_width, _camera_channels);
        _camera.init(_ws_path, 640, 480, 60);

        float camera_width, camera_height, camera_norm_fx, camera_norm_fy, camera_norm_cx, camera_norm_cy;
        _camera.getNormalizedParameters(camera_width, camera_height, camera_norm_fx, camera_norm_fy, camera_norm_cx, camera_norm_cy);
        _camera_fx = camera_norm_fx * _camera_width;
        _camera_fy = camera_norm_fy * _camera_height;
        _camera_cx = camera_norm_cx * _camera_width;
        _camera_cy = camera_norm_cy * _camera_height;
        std::cout << "[DepthCompletion] Camera Intrinsics:" << std::endl;
        std::cout << "[DepthCompletion] Camera Width: "  << _camera_width << std::endl;
        std::cout << "[DepthCompletion] Camera Height: " << _camera_height << std::endl;
        std::cout << "[DepthCompletion] Camera FX: " << _camera_fx << std::endl;
        std::cout << "[DepthCompletion] Camera FY: " << _camera_fy << std::endl;
        std::cout << "[DepthCompletion] Camera CX: " << _camera_cx << std::endl;
        std::cout << "[DepthCompletion] Camera CY: " << _camera_cy << std::endl;

        _num_pixels = _camera_width * _camera_height;

        _X = cv::Mat(_camera_height, _camera_width, CV_32F);
        _Y = cv::Mat(_camera_height, _camera_width, CV_32F);
        for (float i = 0.0f; i < _camera_height; ++i) {
            for (float j = 0.0f; j < _camera_width; ++j) {
                _X.at<float>(i, j) = (j - _camera_cx) / _camera_fx;
                _Y.at<float>(i, j) = (i - _camera_cy) / _camera_fy;
            }
        }
        _X = _X.reshape(1, _num_pixels);
        _Y = _Y.reshape(1, _num_pixels);

        _clock = rclcpp::Clock();
        
        _pub_color = this->create_publisher<sensor_msgs::msg::Image>("rs_camera/color/image_raw", 1);
        _pub_depth = this->create_publisher<sensor_msgs::msg::Image>("rs_camera/depth/image_raw", 1);
        _pub_all_points = this->create_publisher<sensor_msgs::msg::PointCloud2>("rs_camera/cloud/all", 1);
        _pub_selected_points = this->create_publisher<sensor_msgs::msg::PointCloud2>("rs_camera/cloud/selected", 1);
        _pub_selected_points_array = this->create_publisher<std_msgs::msg::Float64MultiArray>("rs_camera/cloud/selected/array", 1);

        _sub_odometry = this->create_subscription<nav_msgs::msg::Odometry>(
            _control_odom_topic, 1, std::bind(&DepthCompletionNodelet::odomCallback, this, std::placeholders::_1));
        _sub_rs_camera_trigger = this->create_subscription<std_msgs::msg::Empty>(
            "camera/trigger", 1, std::bind(&DepthCompletionNodelet::rsCameraCallback, this, std::placeholders::_1));
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
    
    template<typename T>
    void logValues(const std::string &name = "", const T &values = T(), const int &precision = 3) {
        if (name.empty() && values.size() == 0)
            RCLCPP_INFO(this->get_logger(), "");
        else if (values.size() == 0)
            RCLCPP_INFO(this->get_logger(), "%s", name.c_str());
        else {
            std::stringstream ss;
            int sswidth = precision * 2;
            ss << std::fixed << std::setprecision(precision) << name << ": ";
            for (Eigen::Index i = 0; i < values.size(); i++)
                ss << std::setw(sswidth) << std::right << values(i) << " ";
            RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
        }
    }

    template<typename T>
    void logValue(const std::string &name = "", const T &value = T(), const int &precision = 3) {
        std::stringstream ss;
        int sswidth = precision * 2;
        ss << std::fixed << std::setprecision(precision) << name << ": ";
        ss << std::setw(sswidth) << std::right << value << " ";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }

    void logEmpty() {
        RCLCPP_INFO(this->get_logger(), "");
    }

    realsense_camera::RSCamera _camera;
    MDEEngine _mde_engine;

    rclcpp::Clock _clock;
    std::string _world_frame_id;
    std::string _camera_frame_id;
    int _num_pixels;
    cv::Mat _X, _Y;
    double _epsilon;
    int _num_avoid_points;
    std::string _control_odom_topic;
    bool _is_odom_init;

    float _camera_fx;
    float _camera_fy;
    float _camera_cx;
    float _camera_cy;
    int _camera_width;
    int _camera_height;
    int _camera_channels;
    int _camera_fps;
    float _camera_min_range;
    float _camera_max_range;
    int _speckle_max_size;
    int _speckle_diff;
    double _robot_width;
    double _robot_height;

    int _pooling_width;
    int _pooling_height;

    int _engine_width;
    int _engine_height;
    std::string _engine_path;
    std::string _ws_path;
    int _batch_size;

    double _max_obs_rel_pos;
    double _max_obs_rel_vel;
    double _max_obs_rel_acc;

    Eigen::Vector3f _t_W_B;
    Eigen::Vector3f _v_W;
    Eigen::Matrix3f _R_W_B;
    Eigen::Quaternionf _q_B_C;
    Eigen::Quaternionf _q_C_B;
    Eigen::Matrix3f _R_B_C;
    Eigen::Vector3f _t_B_C;
    double _t_W_B_z_0;

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
    void rsCameraCallback(const std_msgs::msg::Empty::SharedPtr empty_msg);
    sensor_msgs::msg::PointCloud2 getCloudMsg(const Eigen::MatrixXf& point_cloud);
    bool isRobotProjectedAtDepth(float depth, int x, int y);
        
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_color;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _pub_depth;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_all_points;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_selected_points;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr _pub_selected_points_array;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _sub_odometry;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr _sub_rs_camera_trigger;
};

void DepthCompletionNodelet::odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {

    if (!_is_odom_init) {
        _t_W_B_z_0 = odom_msg->pose.pose.position.z;
        _is_odom_init = true;
    }

    _t_W_B = Eigen::Vector3f(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
    _v_W = Eigen::Vector3f(odom_msg->twist.twist.linear.x, odom_msg->twist.twist.linear.y, odom_msg->twist.twist.linear.z);
    Eigen::Quaternionf q_W_B(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                             odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
    q_W_B.normalize();
    _R_W_B = q_W_B.toRotationMatrix();
}

void DepthCompletionNodelet::rsCameraCallback(const std_msgs::msg::Empty::SharedPtr empty_msg) {

    // Load frames
    uint64_t timestamp;
    cv::Mat color_image, depth_image;
    if (!_camera.grabFrames(color_image, depth_image, timestamp)) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), _clock, 1000,
            "[RSCamera] Dropped frame longing than RealSense API timeout!");
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

    // Resize frame
    cv::resize(completed_depth, completed_depth, cv::Size(_camera_width, _camera_height), 0, 0, cv::INTER_LINEAR);

    // Convert OpenCV Mat to Torch Tensor
    torch::Tensor completed_depth_tensor = torch::from_blob(
        completed_depth.data, {completed_depth.rows, completed_depth.cols}, torch::kFloat32).clone();
    completed_depth_tensor = completed_depth_tensor.unsqueeze(0).unsqueeze(0);

    // Perform max pooling on the negated input to get min pooling results
    torch::Tensor neg_input, neg_pooled, indices;
    neg_input = -completed_depth_tensor;
    std::tie(neg_pooled, indices) = torch::max_pool2d_with_indices(
        neg_input, {_pooling_width, _pooling_height}, {_pooling_width, _pooling_height}, {0, 0}, {1, 1}, false);
    indices = indices.squeeze();

    // Select the points from the original depth image
    std::vector<Eigen::Vector3f> pooled_points;
    std::vector<float> depths;

    for (int y = 0; y < indices.size(0); ++y) {
        for (int x = 0; x < indices.size(1); ++x) {

            int idx = indices[y][x].item<int>();
            int original_y = idx / completed_depth.cols;
            int original_x = idx % completed_depth.cols;

            // Depth value at the selected point
            float depth_value = completed_depth.at<float>(original_y, original_x);

            // Convert to 3D point using camera intrinsics
            float X = (original_x - _camera_cx) * (depth_value / _camera_fx);
            float Y = (original_y - _camera_cy) * (depth_value / _camera_fy);
            float Z = depth_value;
            Eigen::Vector3f p_C(X, Y, Z);

            // Convert to the world frame
            Eigen::Vector3f p_B = _R_B_C * p_C + _t_B_C;
            Eigen::Vector3f p_W = _R_W_B * p_B + _t_W_B;

            // Check if the point is projected on the image plane
            if (!isRobotProjectedAtDepth(depth_value, original_x, original_y)) {
                continue;
            }

            // Skip points that don't meet odometry initialization condition
            if (_is_odom_init && (p_W(2) - _t_W_B_z_0) <= 0.2f) {
                continue;
            }

            // float projection = _v_W.transpose().cast<float>() * p_W;
            // projection = (projection > _epsilon) ? projection : _epsilon;
            // ttcs(0, i) = depth; // / projection;
            pooled_points.push_back(p_W);
            depths.push_back(depth_value);
        }
    }

    // Publish point cloud
    Eigen::MatrixXf P_W(3, pooled_points.size());
    for (size_t i = 0; i < pooled_points.size(); ++i) {
        P_W.col(i) = pooled_points[i];
    }
    _pub_all_points->publish(getCloudMsg(P_W));

    // Create a vector of indices and sort it based on the depths
    std::vector<int> indices_sorted(depths.size());
    std::iota(indices_sorted.begin(), indices_sorted.end(), 0); // Fill with 0, 1, ..., n-1
    std::partial_sort(indices_sorted.begin(), 
                      indices_sorted.begin() + std::min(_num_avoid_points, static_cast<int>(indices_sorted.size())), 
                      indices_sorted.end(), 
                      [&](int a, int b) { return depths[a] < depths[b]; });

    // Prepare the selected points
    std::vector<Eigen::Vector3f> selected_points;
    for (int i = 0; i < std::min(_num_avoid_points, static_cast<int>(indices_sorted.size())); ++i) {
        selected_points.push_back(pooled_points[indices_sorted[i]]);
    }

    // If we have fewer than _num_avoid_points, fill the rest with fake states
    Eigen::Vector3f fake_state(_t_W_B(0) + _max_obs_rel_pos, _t_W_B(1), _t_W_B(2));
    while (selected_points.size() < _num_avoid_points) {
        selected_points.push_back(fake_state);
    }

    // Publish selected points
    Eigen::MatrixXf selected_points_matrix(3, _num_avoid_points);
    for (size_t i = 0; i < selected_points.size(); ++i) {
        selected_points_matrix.col(i) = selected_points[i];
    }
    _pub_selected_points->publish(getCloudMsg(selected_points_matrix));

    // Convert to std::vector for the Float64MultiArray message
    std::vector<double> selected_points_vector(
        selected_points_matrix.data(), selected_points_matrix.data() + selected_points_matrix.size());
    std_msgs::msg::Float64MultiArray msg;
    msg.data = selected_points_vector;
    _pub_selected_points_array->publish(msg);
}

sensor_msgs::msg::PointCloud2 DepthCompletionNodelet::getCloudMsg(const Eigen::MatrixXf& point_cloud) {

    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = _clock.now();
    cloud_msg.header.frame_id = _camera_frame_id;
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

bool DepthCompletionNodelet::isRobotProjectedAtDepth(float depth, int x, int y) {

    // Return the bounding box of the robot's projection in the image plane
    int width_in_image  = static_cast<int>((_robot_width  * _camera_fx) / depth);
    int height_in_image = static_cast<int>((_robot_height * _camera_fy) / depth);
    cv::Rect robot_projection(_camera_width / 2 - width_in_image / 2, _camera_height / 2 - height_in_image / 2, width_in_image, height_in_image);
    return robot_projection.contains(cv::Point(x, y));
}

}  // namespace arpl_depth_completion

RCLCPP_COMPONENTS_REGISTER_NODE(arpl_depth_completion::DepthCompletionNodelet)