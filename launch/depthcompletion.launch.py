import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration, TextSubstitution, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer, PushRosNamespace
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument('name',           default_value='quadrotor'),
        DeclareLaunchArgument('platform_type',  default_value='race2'),
        DeclareLaunchArgument('control_odom',   default_value='odom'),
    ]

    control_config =  PathJoinSubstitution([
        get_package_share_directory('arpl_autonomy'), TextSubstitution(text='config'), 
        LaunchConfiguration('platform_type'), TextSubstitution(text='default'), TextSubstitution(text='control.yaml')])

    workspace_path = os.path.abspath(os.path.join(get_package_share_directory('arpl_autonomy'), '..', '..', '..', '..'))

    depth_completion_node = ComposableNode(
        package="arpl_depth_completion",
        plugin="arpl_depth_completion::DepthCompletionNodelet",
        namespace=LaunchConfiguration('name'),
        name="depth_completion_nodelet",
        parameters=[control_config,
                    {'quadrotor_name': LaunchConfiguration('name'), 
                     'workspace_path': workspace_path,
                     'control_odom':   LaunchConfiguration('control_odom')}],
    )

    depth_completion_container = ComposableNodeContainer(
        name='depth_completion_container',
        namespace=LaunchConfiguration('name'),
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[depth_completion_node],
        output='screen',
    )

    camera_trigger_process = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--rate', '30', '/race8/camera/trigger', 'std_msgs/msg/Empty', '{}'],
    )

    ld = LaunchDescription(launch_args)
    ld.add_action(depth_completion_container)
    ld.add_action(camera_trigger_process)

    return ld
