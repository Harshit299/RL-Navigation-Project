# ROS 2 Deep Reinforcement Learning Local Planner

An autonomous differential-drive robot local path planner trained using **Deep Reinforcement Learning (PPO)** in a custom Gymnasium environment and deployed on **ROS 2**.

The system leverages an **ONNX-exported Policy Network** to process real-time LiDAR scans and target goal coordinates, executing smooth velocity commands to escape deceptive obstacle geometries, including tight **V-traps** and narrow **gap-traps**, inside a **14 m × 20 m** simulated arena.

---

## ✨ Key Features

- **DRL Motion Planning (PPO):** Trained using Proximal Policy Optimization to avoid local minima, handle deceptive geometries, and execute smooth reverse and turning maneuvers.

- **Custom Python Simulator Node (`env_simulator`):** Simulates differential-drive forward kinematics, vector-based sliding collision physics, 24-beam 360° LiDAR raycasting, and hard boundary constraints.

- **Real-time ONNX Inference (`controller_node`):** Loads the neural network policy (`.onnx`) and executes high-speed control loops without requiring the PyTorch runtime.

- **Custom Obstacles:** Includes V-shaped obstacles, obstacles with narrow gaps, and normal cylindrical obstacles.

- **RViz 3D Visualization:** Full `TF2` coordinate transform broadcasting (`world` → `robot`), `MarkerArray` rendering for blue cylindrical obstacles and red arena walls, and `LaserScan` visualization.

---

## 📂 Project Directory Structure

```text
├── PPO_Brain/                              # DRL Offline Training Pipeline
│   ├── logs/                               # Training execution logs
│   ├── ppo_local_planner_v1/               # Saved model checkpoints
│   ├── ppo_robot_tensorboard/              # TensorBoard event files for monitoring
│   ├── export_onnx.py                      # PyTorch to ONNX policy model converter
│   ├── GUI_Trap_Visualiser.html            # Interactive GUI for viewing trap arrays
│   ├── Gymnasium_Env.py                    # Custom Gymnasium environment registration
│   ├── new_vis.html                        # Supplementary 2D visualization UI
│   ├── robot_env_new.py                    # Core training environment
│   │                                       # (physics & LiDAR raycasting)
│   ├── robot_local_planner.onnx            # Exported trained policy model
│   ├── robot_local_planner.onnx.data       # ONNX model weights
│   ├── Train.py                            # PPO training script (Stable-Baselines3)
│   └── Trap_coord_generator.py             # Generator for V-trap and Gap-trap geometries
│
└── rl_project/                             # ROS 2 Workspace
    └── src/
        └── RL_local_planner/               # ROS 2 Package
            ├── launch/
            │   └── rl.launch.py            # Launch file (Simulator + Controller + RViz)
            │
            ├── RL_local_planner/
            │   ├── __init__.py
            │   ├── env_simulator.py       # Simulation, Raycasting & Physics Node
            │   └── controller.py           # ONNX Inference & Velocity Controller
            │
            ├── Resource/
            │   └── RL_Local_Planner
            │
            ├── test/
            │   ├── test_copyright.py
            │   ├── test_flake8.py
            │   └── test_pep257.py
            │
            ├── package.xml
            ├── setup.cfg
            └── setup.py                    # Console script entry points
│
└── .gitignore
```

---

## 🎥 Demo Video

Watch the complete demonstration of the **ROS 2 Deep Reinforcement Learning Local Planner** on YouTube.

[![ROS 2 Deep Reinforcement Learning Local Planner Demo](https://img.youtube.com/vi/ELjiVeb6hOk/maxresdefault.jpg)](https://youtu.be/ELjiVeb6hOk)

▶️ **[Watch the Demo on YouTube](https://youtu.be/ELjiVeb6hOk)**

The demo shows the trained PPO policy performing local navigation in the simulated environment, including obstacle avoidance and handling deceptive **V-trap** and **gap-trap** configurations.

---
