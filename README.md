# ROS 2 Deep Reinforcement Learning Local Planner

An autonomous differential-drive robot local path planner trained using **Deep Reinforcement Learning (PPO)** in a custom Gymnasium environment and deployed on **ROS 2**. 

The system leverages an **ONNX-exported Policy Network** to process real-time LiDAR scans and target goal coordinates, executing smooth velocity commands to escape deceptive obstacle geometries—including tight **V-traps** and narrow **gap-traps**—inside a $14\text{m} \times 20\text{m}$ simulated arena.

---

## Key Features

* **DRL Motion Planning (PPO):** Trained using Proximal Policy Optimization to avoid local minima, handle deceptive geometries, and execute smooth reverse/turning maneuvers.
* **Custom Python Simulator Node (`env_simulator`):** Simulates differential drive forward kinematics, vector-based sliding collision physics, 24-beam 360° LiDAR raycasting, and hard boundary constraints.
* **Real-time ONNX Inference (`controller_node`):** Loads the neural network policy (`.onnx`) and executes high-speed control loops without PyTorch runtime dependencies.
* **OccupancyGrid Generation:** Automatically rasterizes continuous circular obstacles into a 0.1m-resolution `nav_msgs/msg/OccupancyGrid` map for Nav2 integration.
* **RViz 3D Visualization:** Full `TF2` coordinate transform broadcasting (`world` $\rightarrow$ `robot`), `MarkerArray` rendering for blue cylinder obstacles/red arena walls, and `LaserScan` visualization.

---

## Project Directory Structure

```text
~/
├── PPO_Brain/                        # DRL Offline Training Pipeline
│   ├── logs/                         # Training execution logs
│   ├── ppo_local_planner_v1/         # Saved model checkpoints
│   ├── ppo_robot_tensorboard/        # TensorBoard event files for curve monitoring
│   ├── export_onnx.py                # PyTorch to ONNX policy model converter
│   ├── GUI_Trap_Visualiser.html      # Interactive HTML GUI for viewing trap arrays
│   ├── Gymnasium_Env.py              # Custom Gymnasium environment registration
│   ├── new_vis.html                  # Supplementary 2D visualization UI
│   ├── robot_env_new.py              # Core training environment (physics & raycasting)
│   ├── robot_local_planner.onnx      # Exported trained policy model
│   ├── robot_local_planner.onnx.data # ONNX model weights
│   ├── Train.py                      # Primary PPO training script (Stable-Baselines3)
│   └── Trap_coord_generator.py       # Mathematical generator for V-trap and Gap-trap geometries
│
└── Navigator/                        # ROS 2 Workspace
    └── src/
        └── RL_local_planner/         # ROS 2 Package
            ├── launch/
            │   └── rl.launch.py      # Launch file (Simulator + Controller + RViz)
            ├── RL_local_planner/
            │   ├── __init__.py
            │   ├── env_simulator.py  # Simulation, Raycasting & Physics Node
            │   └── controller.py     # ONNX Inference & Velocity Controller Node
            ├── Resource/
            │   ├──RL_Local_Planner
            ├── test/
            │   ├── test_copyright.py
            │   ├── test_flake8.py  # Simulation, Raycasting & Physics Node
            │   └── test_pep257.py     # ONNX Inference & Velocity Controller Node
            ├── package.xml
            ├── setup.cfg
            └── setup.py              # Console script entry points setup
├── .gitignore
