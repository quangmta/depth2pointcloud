import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    param_config = os.path.join(get_package_share_directory('pointcloud_processing'), 'config', 'param.yaml')
    return LaunchDescription([
        Node
        (
            package='pointcloud_processing',
            executable='pointcloud_processing_node',
            name='pointcloud_processing_node',
            # remappings=[('/image','/camera/color/image_raw'),
            #             ('/depth','/camera/aligned_depth_to_color/image_raw'),
            #             ('/camera_info', '/camera/color/camera_info'),
            #             ('/laser', '/scan')],
            remappings=[('/image','/rgb_image_sync'),
                        ('/depth','/estimated_depth'),
                        ('/camera_info', '/camera/color/camera_info'),
                        ('/laser', '/scan')],
                        
            parameters=[param_config]           
        )
    ])