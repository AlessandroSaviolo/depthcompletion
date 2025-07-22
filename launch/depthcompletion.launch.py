import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

def find_workspace_directory(package_name):
    # Locate the workspace directory for a given package.
    install_dir = get_package_share_directory(package_name)
    return os.path.abspath(os.path.join(install_dir, '..', '..', '..', '..'))

def generate_launch_description():
    # Path to the config file
    config_file = os.path.join(
        get_package_share_directory("depthcompletion"),
        "config",
        "params.yaml"
    )

    # Locate workspace and engine directories
    package_share_directory_path = get_package_share_directory('depthcompletion')
    workspace_path = os.path.abspath(os.path.join(package_share_directory_path, '..', '..', '..', '..'))
    engine_path = os.path.join(workspace_path, '/src/depthcompletion/scripts/depth_anything_v2/checkpoints/depth_anything_v2_vits.trt')

    # Define the depth completion node
    depthcompletion_node = ComposableNode(
        package="depthcompletion",
        plugin="depthcompletion::DepthCompletionNodelet",
        namespace='',
        name="depthcompletion_nodelet",
        parameters=[{'workspace_path':       workspace_path, 
                     'engine.relative_path': engine_path},
                     config_file]
    )

    # Define the container for the composable node
    depthcompletion_container = ComposableNodeContainer(
        name='depthcompletion_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[depthcompletion_node],
        output='screen',
    )

    # Assemble the launch description
    ld = LaunchDescription()
    ld.add_action(depthcompletion_container)

    return ld
