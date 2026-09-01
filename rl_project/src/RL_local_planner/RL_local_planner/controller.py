# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist, PoseStamped
# from visualization_msgs.msg import Marker, MarkerArray
# import numpy as np
# import onnxruntime as ort
# import math

# class HybridController(Node):
#     def __init__(self):
#         super().__init__('controller_node')

#         # Load the trained ONNX Neural Network
#         model_path = "/mnt/d/Python_Engine_2/robot_local_planner.onnx"
#         try:
#             self.onnx_engine = ort.InferenceSession(model_path)
#             self.get_logger().info(f"Successfully loaded RL model: {model_path}")
#         except Exception as e:
#             self.get_logger().error(f"Failed to load ONNX model: {e}")
#             raise e


#         self.robot_x = 0.0
#         self.robot_y = 0.0
#         self.robot_theta = 0.0
        
#         self.goal_x = None
#         self.goal_y = None
        
#         # Initialize lidar with default values (1 = Max distance normalized)
#         self.current_lidar_scan = np.full(24, 1.0, dtype=np.float32)
#         self.max_lidar_range = 5.0

#         self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.pose_subscriber = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
#         self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
#         self.goal_subscriber = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

#         self.timer = self.create_timer(0.05, self.control_loop)

#     def pose_callback(self, msg):
#         self.robot_x = msg.pose.pose.position.x
#         self.robot_y = msg.pose.pose.position.y
#         q = msg.pose.pose.orientation
#         siny_cosp = 2 * (q.w * q.z + q.x * q.y)
#         cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
#         self.robot_theta = math.atan2(siny_cosp, cosy_cosp)

#     def lidar_callback(self, msg):
#         ranges = np.array(msg.ranges, dtype = np.float32)
#         # substract robot radius
#         safe_ranges = np.maximum(ranges - 0.125, 0.0)

#         # Normalize the lidar scans between 0.0 and 1.0 for the neural network
#         self.current_lidar_scan = np.clip(safe_ranges / self.max_lidar_range, 0.0, 1.0)

#     def goal_callback(self, msg):
#         self.goal_x = msg.pose.position.x
#         self.goal_y = msg.pose.position.y

#     def control_loop(self):
#         # If no goal is set, stop the robot
#         if self.goal_x is None or self.goal_y is None:
#             self.velocity_publish(0.0, 0.0)
#             return

#         # Calculate relative distance to goal
#         dist_to_goal = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        
#         # Goal arrival check
#         if dist_to_goal < 0.2:
#             self.get_logger().info("Goal Reached")
#             self.velocity_publish(0.0, 0.0)
#             self.goal_x = None
#             self.goal_y = None
#             return

#         # Calculate angle to goal relative to robot (that's why substracting robot_theta)
#         angle_to_goal = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x) - self.robot_theta

#         # Bounding angle in [-pi, pi]
#         angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi

#         # Map distance to [0, 1] 
#         norm_dist = np.clip(dist_to_goal / 20.0, 0.0, 1.0)

#         # Map angle to [-1, 1]
#         norm_angle = angle_to_goal / math.pi 

#         # [24 LiDAR beams, 1 normalized distance, 1 normalized angle]
#         observation = np.concatenate([self.current_lidar_scan, [norm_dist, norm_angle]])
#         observation = observation.astype(np.float32).reshape(1, -1)

#         # Neural Network Inference
#         try:
#             input_name = self.onnx_engine.get_inputs()[0].name
#             onnx_inputs = {input_name: observation}
#             action_outputs = self.onnx_engine.run(None, onnx_inputs)[0]

#             raw_linear = float(action_outputs[0][0])  # Output is [-1.0, 1.0]
#             raw_angular = float(action_outputs[0][1]) # Output is [-1.0, 1.0]

#             # Scale raw actions to actual physical limits
#             max_linear_vel = 0.5   # 0.5 m/s
#             max_angular_vel = 1.5  # 1.5 rad/s
            
#             # Map linear from [-1, 1] to [0, max_linear_vel] (No backward driving)
#             # linear_v = ((raw_linear + 1.0) / 2.0) * max_linear_vel  #---> with this line the robot won't reverse

#             # [-max_linear_vel, max_linear_vel] so the agent can reverse
#             linear_v = raw_linear * max_linear_vel

#             # Map angular from [-1, 1] to [-max_angular_vel, max_angular_vel]
#             angular_w = raw_angular * max_angular_vel
            
#             self.velocity_publish(linear_v, angular_w)

#         except Exception as e:
#             self.get_logger().error(f"Inference Error: {e}")
#             self.velocity_publish(0.0, 0.0)

#     def velocity_publish(self, linear_v, angular_w):
#         msg = Twist()
#         msg.linear.x = linear_v
#         msg.linear.y = 0.0
#         msg.linear.z = 0.0
#         msg.angular.x = 0.0
#         msg.angular.y = 0.0
#         msg.angular.z = angular_w

#         self.vel_publisher.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = HybridController()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()















#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import onnxruntime as ort
import math
from collections import deque

class HybridController(Node):
    def __init__(self):
        super().__init__('controller_node')

        # Load the trained ONNX Neural Network
        model_path = "/mnt/d/Python_Engine_2/robot_local_planner.onnx"
        try:
            self.onnx_engine = ort.InferenceSession(model_path)
            self.get_logger().info(f"Successfully loaded RL model: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            raise e


        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        self.goal_x = None
        self.goal_y = None
        
        # Initialize lidar with default values (1 = Max distance normalized)
        self.current_lidar_scan = np.full(24, 1.0, dtype=np.float32)
        self.max_lidar_range = 5.0
        # Beam angles in the robot frame, matching the simulator's convention
        # (index 0 = straight ahead, evenly spaced counter-clockwise)
        self.lidar_beam_angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)

        # ---- Stuck detection / recovery state ----
        # Rolling window of recent (x, y) positions used to detect "no progress"
        self.pose_history = deque(maxlen=40)   # 40 ticks * 0.05s = 2.0s window
        self.stuck_distance_threshold = 0.08   # meters of net movement expected per window
        self.recovery_active = False
        self.recovery_start_time = None
        self.max_recovery_time = 4.0           # safety cap so recovery can't run forever

        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_subscriber = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.goal_subscriber = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.timer = self.create_timer(0.05, self.control_loop)

    def pose_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges, dtype = np.float32)
        # substract robot radius
        safe_ranges = np.maximum(ranges - 0.125, 0.0)

        # Normalize the lidar scans between 0.0 and 1.0 for the neural network
        self.current_lidar_scan = np.clip(safe_ranges / self.max_lidar_range, 0.0, 1.0)

    def goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

    def control_loop(self):
        # If a recovery maneuver is in progress, let it finish before
        # handing control back to the learned policy
        if self.recovery_active:
            self.run_recovery_behavior()
            return

        # If no goal is set, stop the robot
        if self.goal_x is None or self.goal_y is None:
            self.velocity_publish(0.0, 0.0)
            self.pose_history.clear()
            return

        # Calculate relative distance to goal
        dist_to_goal = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        
        # Goal arrival check
        if dist_to_goal < 0.2:
            self.get_logger().info("Goal Reached")
            self.velocity_publish(0.0, 0.0)
            self.goal_x = None
            self.goal_y = None
            self.pose_history.clear()
            return

        # ---- Stuck detection ----
        # The learned policy is purely reactive (no memory of past states), so
        # concave obstacle clusters can trap it in a local minimum: every
        # locally-preferred action still points back into the pocket. Track
        # net displacement over a rolling time window and trigger a rule-based
        # recovery behavior if the robot isn't actually making progress.
        self.pose_history.append((self.robot_x, self.robot_y))
        if len(self.pose_history) == self.pose_history.maxlen:
            x0, y0 = self.pose_history[0]
            net_movement = math.hypot(self.robot_x - x0, self.robot_y - y0)
            if net_movement < self.stuck_distance_threshold:
                self.get_logger().warn(
                    f"Robot appears stuck (moved {net_movement:.3f}m in "
                    f"{self.pose_history.maxlen * 0.05:.1f}s) - starting recovery"
                )
                self.start_recovery()
                self.run_recovery_behavior()
                return

        # Calculate angle to goal relative to robot (that's why substracting robot_theta)
        angle_to_goal = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x) - self.robot_theta

        # Bounding angle in [-pi, pi]
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi

        # Map distance to [0, 1] 
        norm_dist = np.clip(dist_to_goal / 20.0, 0.0, 1.0)

        # Map angle to [-1, 1]
        norm_angle = angle_to_goal / math.pi 

        # [24 LiDAR beams, 1 normalized distance, 1 normalized angle]
        observation = np.concatenate([self.current_lidar_scan, [norm_dist, norm_angle]])
        observation = observation.astype(np.float32).reshape(1, -1)

        # Neural Network Inference
        try:
            input_name = self.onnx_engine.get_inputs()[0].name
            onnx_inputs = {input_name: observation}
            action_outputs = self.onnx_engine.run(None, onnx_inputs)[0]

            raw_linear = float(action_outputs[0][0])  # Output is [-1.0, 1.0]
            raw_angular = float(action_outputs[0][1]) # Output is [-1.0, 1.0]

            # Scale raw actions to actual physical limits
            max_linear_vel = 0.5   # 0.5 m/s
            max_angular_vel = 1.5  # 1.5 rad/s
            
            # Map linear from [-1, 1] to [0, max_linear_vel] (No backward driving)
            # linear_v = ((raw_linear + 1.0) / 2.0) * max_linear_vel  #---> with this line the robot won't reverse

            # [-max_linear_vel, max_linear_vel] so the agent can reverse
            linear_v = raw_linear * max_linear_vel

            # Map angular from [-1, 1] to [-max_angular_vel, max_angular_vel]
            angular_w = raw_angular * max_angular_vel
            
            self.velocity_publish(linear_v, angular_w)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")
            self.velocity_publish(0.0, 0.0)

    def start_recovery(self):
        """Begin a rule-based escape maneuver. Called once when the stuck
        detector fires."""
        self.recovery_active = True
        self.recovery_start_time = self.get_clock().now()
        self.pose_history.clear()

    def run_recovery_behavior(self):
        """Turn in place toward the most open lidar direction, then drive
        forward once roughly facing it. Re-evaluates the open direction every
        tick since new lidar data keeps arriving. This intentionally does not
        reverse blindly, since in a tight pocket the space behind the robot
        may be just as blocked as the space ahead."""
        elapsed = (self.get_clock().now() - self.recovery_start_time).nanoseconds * 1e-9

        open_beam_idx = int(np.argmax(self.current_lidar_scan))
        open_range = float(self.current_lidar_scan[open_beam_idx])
        open_angle = self.lidar_beam_angles[open_beam_idx]
        # Wrap into [-pi, pi] so we know the shortest way to turn
        open_angle = (open_angle + math.pi) % (2 * math.pi) - math.pi

        # Give up and hand back control if: we've spent too long recovering,
        # or even the "most open" direction has become tight (fully boxed in) -
        # in that case further blind turning won't help.
        if elapsed > self.max_recovery_time or open_range < 0.10:
            self.get_logger().info("Recovery finished, handing back to policy")
            self.recovery_active = False
            self.pose_history.clear()
            self.velocity_publish(0.0, 0.0)
            return

        if abs(open_angle) > 0.35:
            # Rotate in place toward the opening first (no forward motion,
            # so it's safe even if the robot is wedged front and back)
            turn_dir = 1.0 if open_angle > 0 else -1.0
            self.velocity_publish(0.0, turn_dir * 0.9)
        else:
            # Roughly facing the opening now - drive out through it
            self.velocity_publish(0.15, open_angle)

    def velocity_publish(self, linear_v, angular_w):
        msg = Twist()
        msg.linear.x = linear_v
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_w

        self.vel_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HybridController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()