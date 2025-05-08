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
    # Declare launch arguments
    launch_args = [
        DeclareLaunchArgument('name',                   default_value='quadrotor'),
        DeclareLaunchArgument('engine.width',           default_value='322'),
        DeclareLaunchArgument('engine.height',          default_value='238'),
        DeclareLaunchArgument('engine.batchsize',       default_value='1'),
        DeclareLaunchArgument('camera.frame_id',        default_value='camera_link'),
        DeclareLaunchArgument('camera.width',           default_value='640'),
        DeclareLaunchArgument('camera.height',          default_value='480'),
        DeclareLaunchArgument('camera.channels',        default_value='3'),
        DeclareLaunchArgument('camera.fps',             default_value='60'),
        DeclareLaunchArgument('camera.min_range',       default_value='0.6'),
        DeclareLaunchArgument('camera.max_range',       default_value='10.0'),
        DeclareLaunchArgument('camera.speckle_max_size',default_value='500'),
        DeclareLaunchArgument('camera.speckle_diff',    default_value='200')
    ]

    # Locate workspace and engine directories
    package_share_directory_path = get_package_share_directory('depthcompletion')
    workspace_path = os.path.abspath(os.path.join(package_share_directory_path, '..', '..', '..', '..'))
    engine_path = os.path.join(workspace_path, '/src/depthcompletion/scripts/depth_anything_v2/checkpoints/depth_anything_v2_vits.trt')

    # Define the depth completion node
    depthcompletion_node = ComposableNode(
        package="depthcompletion",
        plugin="depthcompletion::DepthCompletionNodelet",
        namespace=LaunchConfiguration('name'),
        name="depthcompletion_nodelet",
        parameters=[{'engine.width':            LaunchConfiguration('engine.width'),
                     'engine.height':           LaunchConfiguration('engine.height'),
                     'engine.batchsize':        LaunchConfiguration('engine.batchsize'),
                     'camera.frame_id':         LaunchConfiguration('camera.frame_id'),
                     'camera.width':            LaunchConfiguration('camera.width'),
                     'camera.height':           LaunchConfiguration('camera.height'),
                     'camera.channels':         LaunchConfiguration('camera.channels'),
                     'camera.fps':              LaunchConfiguration('camera.fps'),
                     'camera.min_range':        LaunchConfiguration('camera.min_range'),
                     'camera.max_range':        LaunchConfiguration('camera.max_range'),
                     'camera.speckle_max_size': LaunchConfiguration('camera.speckle_max_size'),
                     'camera.speckle_diff':     LaunchConfiguration('camera.speckle_diff'),
                     'workspace_path':          workspace_path, 
                     'engine.relative_path':    engine_path}]
    )

    # Define the container for the composable node
    depthcompletion_container = ComposableNodeContainer(
        name='depthcompletion_container',
        namespace=LaunchConfiguration('name'),
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[depthcompletion_node],
        output='screen',
    )

    # Assemble the launch description
    ld = LaunchDescription(launch_args)
    ld.add_action(depthcompletion_container)

    return ld
