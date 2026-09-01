import gymnasium as gym
from typing import Optional
import numpy as np
from collections import deque

class RobotEscapeEnv(gym.Env):

    def __init__(self):
        super(RobotEscapeEnv, self).__init__()

        # =========== define observation and action spaces ==================
        
        # 24 lidar data and 2d goal pose (distance to goal and angle to goal) - normalised between 0 and 1
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (26,), dtype = np.float32)
        self.action_space = gym.spaces.Box(low = np.array([-0.2, -1.0]), # max -ve linear speed, max -ve angular speed
                                           high = np.array([1.5, 1.0]), # max +ve linear speed, max +ve angular speed
                                           dtype = np.float32)
        
        # ===================================================================
        
        self.stagnation_penalty = 0.0
        self.dt = 0.1 # time diff b/w state 1 and state 2
        self.robot_pose = np.array([0.0, 0.0, 0.0]) # x,y,theta
        self.robot_radius = 0.15
        self.obstacle_radius = 0.30
        self.goal_pose = np.array([0.0, 0.0]) # x,y
        self.obstacles = []
        self.prev_dist_to_goal = 0.0
        self.pose_history = deque(maxlen = 30)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)

        self.robot_pose = np.array([self.np_random.uniform(-2.0, 2.0),
                                    self.np_random.uniform(-2.0, 2.0),
                                    self.np_random.uniform(-np.pi, np.pi)])
        
        ''' 
        Generate random obstacle - 5 obstacles with x,y (2) coords. 
        These x,y coords can have numbers b/w -3 and +3
        '''
        self.obstacles = self.np_random.uniform(-3.0, 3.0, size=(5, 2))
 
        # target location (goal pose) should be different from robot pose
        self.goal_pose = np.array([self.np_random.uniform(-2.0, 2.0),
                                   self.np_random.uniform(-2.0, 2.0)])
        
        # Ensure goal is not too close to the robot's starting position
        while np.linalg.norm(self.robot_pose[:2] - self.goal_pose) < 0.5:
            self.goal_pose = np.array([self.np_random.uniform(-2.0, 2.0),
                                       self.np_random.uniform(-2.0, 2.0)])

        self.prev_dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pose)

        self.pose_history.clear() # clear buffer for next episode

        observation = self.get_obs()
        info = self.get_info()

        return observation, info
    
    def compute_reward(self, lidar_scans, cur_dist_to_goal, terminated, stagnation_penalty):
        
        # Progress component
        delta_dist = self.prev_dist_to_goal - cur_dist_to_goal
        reward = -0.01 
        reward += delta_dist * 15.0
        self.prev_dist_to_goal = cur_dist_to_goal
        
        # Crash rule
        if terminated and cur_dist_to_goal > 0.2:
            return -150.0
            
        # Success rule
        if cur_dist_to_goal < 0.2:
            return 200.0 
            
        # Danger zone warning
        if np.min(lidar_scans) < 0.15:
            reward -= 2.0  # Punish getting too close to walls

        reward += stagnation_penalty  # Add stagnation penalty
            
        return reward
    
    def step(self, action):

        truncated = False
        terminated = False

        linear_vel = action[0]
        angular_vel = action[1]

        # kinematics equations for differential drive robot
        self.robot_pose[0] += linear_vel * np.cos(self.robot_pose[2]) * self.dt
        self.robot_pose[1] += linear_vel * np.sin(self.robot_pose[2]) * self.dt
        self.robot_pose[2] += angular_vel * self.dt

        dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pose)
        reached_target = dist_to_goal < 0.20

        lidar_scans = self.simulate_lidar()

        self.pose_history.append(self.robot_pose[:2].copy()) # [:2] means extracting only x and y coords not theta

        if (len(self.pose_history) == 30):

            dist_to_past = np.linalg.norm(self.robot_pose[:2] - self.pose_history[0])
            if dist_to_past < 0.15:
                self.stagnation_penalty = -0.5
            else: self.stagnation_penalty = 0.0


        dist_to_obstacle = np.linalg.norm(self.obstacles - self.robot_pose[:2], axis=1)
        is_collided = np.any(dist_to_obstacle < (self.robot_radius + self.obstacle_radius))

        if (reached_target or is_collided):
            terminated = True

        reward = self.compute_reward(lidar_scans, dist_to_goal, terminated, self.stagnation_penalty)

        observation = self.get_obs()
        return observation, reward, terminated, truncated, {}

    def simulate_lidar(self):
        """
        Simulates 24 LiDAR beams. 
        Implements raycasting math against self.obstacles.
        """
        num_lidar_beams = 24
        lidar_range = 5.0
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
                
                # Get the closest (X, Y) point
                closest_point_on_ray = projection_dist * ray_direction
                perpendicular_dist = np.linalg.norm(vector_to_obstacle - closest_point_on_ray)

                # detecting whether ray hit the obstacle or not
                if perpendicular_dist < self.obstacle_radius:
                    hit_distance = projection_dist - np.sqrt(self.obstacle_radius**2 - perpendicular_dist**2)
                    
                    if 0.0 < hit_distance < min_hit_dist:
                        min_hit_dist = hit_distance

            # Normalize the final recorded distance between 0 and 1
            lidar_scans[ray_idx] = np.clip(min_hit_dist / lidar_range, 0.0, 1.0)
            
        return lidar_scans

    def get_obs(self):
        """Packages the 26 inputs expected by the observation space."""
        lidar = self.simulate_lidar()
        
        dist_to_goal = np.linalg.norm(self.robot_pose[:2] - self.goal_pose)
        angle_to_goal = np.arctan2(self.goal_pose[1] - self.robot_pose[1], 
                                   self.goal_pose[0] - self.robot_pose[0]) - self.robot_pose[2]
        
        # Normalize the relative angle to [-pi, pi]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        # Normalize inputs mathematically to [0.0, 1.0] for the neural network
        norm_dist = np.clip(dist_to_goal / 5.0, 0.0, 1.0) # Assuming an arbitrary 5m max distance
        norm_angle = (angle_to_goal + np.pi) / (2 * np.pi)
        
        return np.concatenate([lidar, [norm_dist, norm_angle]]).astype(np.float32)

    def get_info(self):
        """Returns extra diagnostic information for monitoring training."""
        return {"distance_to_goal": np.linalg.norm(self.robot_pose[:2] - self.goal_pose)}