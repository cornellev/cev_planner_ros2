from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os


def get_path(package, dir, file):
    return os.path.join(get_package_share_directory(package), dir, file)


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="cev_planner_ros2",
                executable="planner_node",
                name="cev_planner_ros2_node",
                output="screen",
                parameters=[
                    get_path("cev_planner_ros2", "config", "cev_planner.yaml")
                ],
            ),
        ]
    )
