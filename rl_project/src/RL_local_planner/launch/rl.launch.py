from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        Node(
            package = 'rviz2',
            executable = 'rviz2',
            output = 'screen'
        ),
        Node(
            package = 'RL_local_planner',
            executable = 'controller_node',
            output = 'screen'
        ),
        Node(
            package = 'RL_local_planner',
            executable = 'simulator_node',
            output = 'screen'
        )
    ])