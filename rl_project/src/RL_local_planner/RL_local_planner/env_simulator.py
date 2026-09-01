#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import math

class LocalPlanner(Node):
    def __init__(self):
        super().__init__('simulator_node')

        self.room_boundaries = [0.0, 0.0, 14.0, 20.0] # [xmin, ymin, xmax, ymax]

        self.GAP_TRAPS = [
            [[7.809, 1.534], [8.191, 0.866]],  # Trap 1
            [[10.41, 0.826], [10.59, 1.574]],  # Trap 2
            [[12.84, 0.85],  [13.16, 1.55]],   # Trap 3
            [[7.844, 3.248], [8.156, 3.952]],  # Trap 4
            [[10.629,3.237], [10.371,3.963]],  # Trap 5
            [[13.38, 3.66],  [12.62, 3.54]],   # Trap 6
            [[8.35,  5.84],  [7.65,  6.16]],   # Trap 7
            [[10.599,5.628], [10.401,6.372]],  # Trap 8
            [[13.294,6.249], [12.706,5.751]],  # Trap 9
            [[7.754, 8.104], [8.246, 8.696]],  # Trap 10
            [[10.601,8.028], [10.399,8.772]],  # Trap 11
            [[13.257,8.113], [12.743,8.687]],  # Trap 12
            [[8.37,  8.695], [7.63,  8.905]],  # Trap 13
            [[10.416,9.176], [10.584,8.424]],  # Trap 14
            [[12.88, 8.434], [13.12, 9.166]],  # Trap 15
        ]

        self.V_TRAPS = [
            [[8.300, 11.214], [7.801, 11.185], [7.302, 11.156], [8.185, 11.701], [8.069, 12.187]],
            [[10.600, 11.214], [10.437, 11.687], [10.273, 12.159], [11.037, 11.457], [11.474, 11.700]],
            [[12.700, 11.214], [13.102, 10.917], [13.503, 10.619], [12.477, 10.767], [12.253, 10.320]],
            [[8.300, 13.761], [8.712, 14.044], [9.125, 14.327], [8.650, 13.404], [9.000, 13.047]],
            [[10.600, 13.761], [10.246, 13.408], [9.892, 13.055], [10.191, 14.049], [9.782, 14.336]],
            [[12.700, 13.761], [12.345, 13.409], [11.991, 13.056], [12.291, 14.049], [11.882, 14.337]],
            [[8.300, 16.380], [7.837, 16.191], [7.375, 16.001], [8.033, 16.803], [7.766, 17.226]],
            [[10.600, 16.380], [10.116, 16.506], [9.632, 16.632], [10.640, 16.878], [10.680, 17.377]],
            [[12.700, 16.380], [12.610, 15.888], [12.520, 15.396], [12.200, 16.383], [11.700, 16.387]],
            [[8.300, 18.857], [8.074, 18.411], [7.848, 17.965], [7.822, 19.002], [7.343, 19.147]],
            [[10.600, 18.857], [10.267, 19.230], [9.934, 19.603], [10.909, 19.250], [11.219, 19.643]],
            [[12.700, 18.857], [13.195, 18.784], [13.689, 18.712], [12.714, 18.357], [12.728, 17.858]],
        ]

        self.random_obs = np.random.uniform(low=[0.5, 0.5], high=[6.5, 19.5], size=(10, 2))

        self.scan_publisher = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10)
        self.visuals_publisher = self.create_publisher(MarkerArray, '/visuals', 10)
        self.vel_subscriber = self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)

        self.timer = self.create_timer(0.05, self.physics_loop)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.robot_pose = np.array([0.0, 0.0, 0.0]) # x,y,theta
        self.robot_radius = 0.125
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        all_obs = []
        for trap in self.GAP_TRAPS:
            all_obs.extend(trap)
        for trap in self.V_TRAPS:
            all_obs.extend(trap)
        all_obs.extend(self.random_obs)
        self.obstacles = np.array(all_obs)
        self.obstacle_radius = 0.25
        self.robot_pose = np.array([2.0, 2.0, 0.0])
        self.dt = 0.1

    def vel_callback(self, msg):

        # self.linear_vel = math.sqrt((msg.linear.x)**2 + (msg.linear.y)**2)
        self.linear_vel = msg.linear.x 
        self.angular_vel = msg.angular.z

    # Helper function to convert yaw angle to ROS 2 Quaternion
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def physics_loop(self):

        # ======== Odometry data =================================
        msg1 = Odometry()
        msg1.header.stamp = self.get_clock().now().to_msg()
        msg1.header.frame_id = "world"
        msg1.child_frame_id = "robot"

        # kinematics equations for differential drive robot
        # self.robot_pose[0] += self.linear_vel * np.cos(self.robot_pose[2]) * self.dt
        # self.robot_pose[1] += self.linear_vel * np.sin(self.robot_pose[2]) * self.dt
        # self.robot_pose[2] += self.angular_vel * self.dt
        # self.robot_pose[2] = (self.robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

        # msg1.pose.pose.position.x = self.robot_pose[0]
        # msg1.pose.pose.position.y = self.robot_pose[1]

        self.robot_pose[0] += self.linear_vel * np.cos(self.robot_pose[2]) * self.dt
        self.robot_pose[1] += self.linear_vel * np.sin(self.robot_pose[2]) * self.dt
        self.robot_pose[2] += self.angular_vel * self.dt
        self.robot_pose[2] = (self.robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

        min_safe_dist = self.robot_radius + self.obstacle_radius
        for obs in self.obstacles:
            vec_to_robot = self.robot_pose[:2] - obs
            dist = np.linalg.norm(vec_to_robot)
            if dist < min_safe_dist:
                overlap = min_safe_dist - dist
                push_direction = vec_to_robot / dist
                self.robot_pose[:2] += push_direction * overlap

        # Hard boundary clipping to simulate rigid walls
        self.robot_pose[0] = np.clip(self.robot_pose[0], self.robot_radius, 14.0 - self.robot_radius)
        self.robot_pose[1] = np.clip(self.robot_pose[1], self.robot_radius, 20.0 - self.robot_radius)

        msg1.pose.pose.position.x = self.robot_pose[0]
        msg1.pose.pose.position.y = self.robot_pose[1]
        msg1.pose.pose.orientation = self.get_quaternion_from_euler(0, 0, self.robot_pose[2])

        # =========================================================

        # ================ Lidar data =============================

        """
        Simulates 24 LiDAR beams. 
        Implements raycasting math against self.obstacles.
        """
        msg2 = LaserScan()
        msg2.header.stamp = self.get_clock().now().to_msg()
        msg2.header.frame_id = "robot"

        num_lidar_beams = 24
        lidar_range = 5.0
        # initialize lidar_scans with 1s
        lidar_scans = np.full(num_lidar_beams, 1.0, dtype=np.float32)
        _, _, theta = self.robot_pose
        angles = np.linspace(0, 2*np.pi, num_lidar_beams, endpoint=False)

        for ray_idx in range(num_lidar_beams):
            # Angle of the ray wrt the world
            ray_angle = angles[ray_idx]

            ray_direction = np.array([np.cos(theta + ray_angle), 
                                      np.sin(theta + ray_angle)])
            
            # Track the closest obstacle hit by this specific ray
            min_hit_dist = lidar_range 
            
            # Checking every ray against each obstacle
            for obs in self.obstacles:
                vector_to_obstacle = obs - self.robot_pose[:2]
                
                # projection_dist: how far along the ray the obstacle sits
                projection_dist = np.dot(vector_to_obstacle, ray_direction)
                
                # If projection is negative, the obstacle is physically behind the laser
                if projection_dist <= 0:
                    continue
                
                # Get the closest (x,y) point
                closest_point_on_ray = projection_dist * ray_direction
                perpendicular_dist = np.linalg.norm(vector_to_obstacle - closest_point_on_ray)

                # detecting whether a ray hits the obstacle or not
                if perpendicular_dist < self.obstacle_radius:
                    hit_distance = projection_dist - np.sqrt(self.obstacle_radius**2 - perpendicular_dist**2)
                    
                    if 0.0 < hit_distance < min_hit_dist:
                        min_hit_dist = hit_distance

            # Normalize the final distance between 0 & 1
            lidar_scans[ray_idx] = float(np.clip(min_hit_dist, 0.0, lidar_range))

        msg2.ranges = lidar_scans.tolist()

        # =========================================================

        # ======== BROADCAST COORDINATE TRANSFORM (TF) =================
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'robot'
        
        t.transform.translation.x = self.robot_pose[0]
        t.transform.translation.y = self.robot_pose[1]
        t.transform.translation.z = 0.0
        t.transform.rotation = self.get_quaternion_from_euler(0, 0, self.robot_pose[2])
        
        # Send the transform over the network to RViz
        self.tf_broadcaster.sendTransform(t)

        # ============== Obstacles Marker data ==============================
        marker_array = MarkerArray()
        self.marker_id_counter = 0 # Reset counter every frame

        # Wall marker
        wall_marker = Marker()
        wall_marker.header.frame_id = "world"
        wall_marker.id = self.marker_id_counter
        self.marker_id_counter += 1
        wall_marker.type = Marker.LINE_STRIP
        wall_marker.action = Marker.ADD
        wall_marker.pose.orientation.w = 1.0
        wall_marker.points = [
            Point(x=0.0, y=0.0), Point(x=14.0, y=0.0), 
            Point(x=14.0, y=20.0), Point(x=0.0, y=20.0), Point(x=0.0, y=0.0)
        ]        
        wall_marker.scale.x = 0.05
        wall_marker.color.a = 1.0
        wall_marker.color.r = 1.0 # Red Walls
        marker_array.markers.append(wall_marker)

        # Obstacle markers
        for trap in self.GAP_TRAPS + self.V_TRAPS:
            for pt in trap:
                obs_marker = Marker()
                obs_marker.header.frame_id = "world"
                obs_marker.id = self.marker_id_counter
                self.marker_id_counter += 1
                obs_marker.type = Marker.CYLINDER 
                obs_marker.action = Marker.ADD

                obs_marker.pose.position.x = pt[0]
                obs_marker.pose.position.y = pt[1]
                obs_marker.pose.position.z = 0.10
                
                # Diameter is radius * 2 (0.25 * 2 = 0.50)
                obs_marker.scale.x = 0.50
                obs_marker.scale.y = 0.50
                obs_marker.scale.z = 0.50
                
                obs_marker.color.a = 1.0
                obs_marker.color.b = 1.0 # Blue Obstacles
                
                marker_array.markers.append(obs_marker)

        for pt in self.random_obs:
            obs_marker = Marker()
            obs_marker.header.frame_id = "world"
            obs_marker.id = self.marker_id_counter
            self.marker_id_counter += 1
            obs_marker.type = Marker.CYLINDER 
            obs_marker.action = Marker.ADD

            obs_marker.pose.position.x = float(pt[0])
            obs_marker.pose.position.y = float(pt[1])
            obs_marker.pose.position.z = 0.10
            
            obs_marker.scale.x = 0.50
            obs_marker.scale.y = 0.50
            obs_marker.scale.z = 0.50
            
            obs_marker.color.a = 1.0
            obs_marker.color.b = 1.0 
            
            marker_array.markers.append(obs_marker)

        # ============================================================

        # ================= Robot Marker =============================
        robot_marker = Marker()
        robot_marker.header.frame_id = "world"
        robot_marker.id = self.marker_id_counter
        robot_marker.type = Marker.CYLINDER
        robot_marker.action = Marker.ADD

        robot_marker.pose.position.x = msg1.pose.pose.position.x
        robot_marker.pose.position.y = msg1.pose.pose.position.y
        robot_marker.pose.position.z = 0.10
        # robot_marker.pose.orientation = msg1.pose.pose.orientation
        robot_marker.pose.orientation = t.transform.rotation
        robot_marker.scale.x = 0.25
        robot_marker.scale.y = 0.25
        robot_marker.scale.z = 0.10
        robot_marker.color.a = 1.0
        robot_marker.color.g = 1.0 # Green Robot
        marker_array.markers.append(robot_marker)

        # ===========================================================
            
        self.odom_publisher.publish(msg1) 
        self.scan_publisher.publish(msg2)      
        self.visuals_publisher.publish(marker_array) 


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()