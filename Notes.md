# RL Complete Picture

* Generally, RL problems are modelled as MDP.

* MDP: Markov Decision Process is the mathematical formulation of RL problems where the future state of the environment depends on the current state and action, NOT on past states.

* There is an agent which learns through trial and error, takes an action, the environment's state changes, and the environment provides it rewards based on whether it is progressing towards the goal or not.

* The goal of RL is to update the policy so that the reward can be maximized.

* Gymnasium is a framework which lets us create the environment for the problem by defining initial states, current values, rules, reward function formulation, etc.

* Stable Baselines3 is a library which contains various algorithms to solve RL problems like PPO, A2C, DQN, SAC, Q-learning, etc.

* While we run the training script, before calling the `step()` function, the `reset()` function is called, which generates random values of x, y and theta for the robot pose. These values are then fed into the `step()` function.

---

## 1. Environment Script and Training Script

The script where we create the environment is purely the blueprint. It contains the class definition, the physics, the rules, and the reward math. We just define how the world works.

We use `gym.make()` in the training script, not in the script where we created the environment.

### 2. Gymnasium

Gymnasium cannot "make" an environment unless it already knows that it exists.

---

## 3. Types of Reward in the RL Project

In the RL project, there are two types of reward:

### 1. Continuous Rewards (rewards earned while running)

These are given when the robot hits the wall/obstacle or reaches the goal. These rewards are given by the `return` statement inside the `if` condition.

### 2. Terminal Rewards (rewards earned when the run ends)

These are given when the robot is continuously reaching the goal, is about to hit the wall/obstacle, or is stuck at some location. These rewards are given using `-=` and `+=`.

---

## 4. Normalized Input and Output

1. RL agent takes normalized input vectors and outputs normalized vectors.

2. We need to convert those normalized output vectors into physical values.

3.

```python
norm_angle = (angle_to_goal + math.pi) / (2 * math.pi)
```

This converts the `angle_to_goal` from `[-π, π]` into the range `[0.0, 1.0]`.

4.

```python
norm_angle = angle_to_goal / math.pi
```

This converts the `angle_to_goal` from `[-π, π]` into the range `[-1.0, 1.0]`.
