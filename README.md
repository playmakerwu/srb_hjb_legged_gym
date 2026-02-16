# SRB-HJB Reinforcement Learning for Legged Locomotion

**Author:** Yiru Wu, University of Wisconsin-Madison

A reinforcement learning framework for quadruped locomotion that replaces conventional position-based control with **direct force control** and augments standard PPO with a **Hamilton-Jacobi-Bellman (HJB) residual loss** derived from a **Single Rigid Body (SRB) dynamics model**. Built on top of NVIDIA Isaac Gym and the legged_gym framework.

---

## Table of Contents

- [Key Idea](#key-idea)
- [Algorithm Overview](#algorithm-overview)
  - [Force-Based Control](#force-based-control)
  - [HJB-Augmented Critic Loss](#hjb-augmented-critic-loss)
  - [SRB Dynamics Model](#srb-dynamics-model)
  - [Full Training Objective](#full-training-objective)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Install Isaac Gym](#step-1--install-isaac-gym)
  - [Step 2 — Clone This Repository and Load Submodules](#step-2--clone-this-repository-and-load-submodules)
  - [Step 3 — Install the RL Library (rsl_rl) from the SRB Branch](#step-3--install-the-rl-library-rsl_rl-from-the-srb-branch)
  - [Step 4 — Install This Package](#step-4--install-this-package)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
- [Toggling SRB-HJB](#toggling-srb-hjb)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Key Idea

Standard RL for legged robots typically uses **position targets** as actions, which are then tracked by low-level PD controllers. This project makes two fundamental changes:

1. **Actions are ground reaction forces (GRFs).** The policy directly outputs desired contact forces in the body frame, which are mapped to joint torques via the contact Jacobian. This gives the policy direct control over the robot's dynamics.

2. **The critic is regularized by physics.** Instead of learning the value function purely from reward signals, we inject a physics-based constraint: the value function should approximately satisfy the HJB equation, where the system dynamics come from a Single Rigid Body model.

---

## Algorithm Overview

### Force-Based Control

In the standard position-control setup, the policy outputs joint position offsets and a PD controller tracks them:

```
τ = Kp * (a * scale + q_default - q) - Kd * dq
```

In our force-control setup (`control_type = "F"`), the policy outputs desired ground reaction forces for each foot in the body frame. These are converted to joint torques via:

```
τ = Σ_feet  (GRF_world)^T · J_foot
```

where `J_foot` is the contact Jacobian and `GRF_world = R_wb * (a * scale + bias)`.

### HJB-Augmented Critic Loss

In continuous-time RL, the optimal value function `V*(x)` satisfies the Hamilton-Jacobi-Bellman equation:

```
ρ V(x) = r(x) + ∇V(x) · f(x)
```

where `ρ = -ln(γ)` is the continuous-time discount rate, `r(x)` is the instantaneous reward, and `f(x)` is the system dynamics.

We enforce this as a soft constraint on the critic network. At each update step, the HJB residual loss is:

```
L_hjb = || ρ V(x) - [ r(x) + ∇_x V(x) · f(x) ] ||²
```

The gradient `∇_x V(x)` is computed by differentiating the critic output with respect to its input via `torch.autograd.grad`, with `create_graph=True` so that the HJB loss itself is differentiable and can backpropagate through the critic.

### SRB Dynamics Model

The dynamics `f(x)` in the HJB equation are provided by a **Single Rigid Body (SRB)** model evaluated inside the simulator at each timestep. The SRB model computes three derivative quantities in the body frame:

```
f(x) = [ base_lin_vel_dot,  base_ang_vel_dot,  projected_gravity_dot ]
```

Specifically:

- **Linear acceleration:** `-ω × v + (1/m) Σ F_contact + g_projected`
- **Angular acceleration:** `I⁻¹ Σ (r_foot × F_contact)`
- **Gravity derivative:** `-ω × g_projected`

where contact forces are rotated into the body frame, foot positions are expressed relative to the base, and the inertia tensor `I` accounts for both the base link and motor masses.

These SRB derivatives are stored in `srb_dynamics_buf` (shape `[num_envs, 9]`) and passed to the PPO update as the ground-truth `f(x)` for the HJB loss.

### Full Training Objective

The total loss for each PPO update is:

```
L = L_surrogate + c_v * L_value - c_e * H(π) + c_hjb * L_hjb
```

| Term | Description |
|------|-------------|
| `L_surrogate` | Clipped PPO surrogate objective (policy loss) |
| `L_value` | Clipped value function regression loss |
| `H(π)` | Policy entropy bonus |
| `L_hjb` | HJB residual loss (physics-informed critic regularization) |
| `c_hjb` | HJB loss coefficient (`hjb_coef`, default 0.1) |

When `hjb_coef = 0` or `enable_srb_dynamics = False`, the system reduces to standard PPO.

---

## Project Structure

```
legged_gym/
├── README.md
├── setup.py
├── .gitmodules                          # submodule registration
│
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── legged_robot.py          # main env: init, step, reset, buffers
│   │   │   ├── legged_robot_config.py   # all config dataclasses
│   │   │   ├── legged_robot_rewards.py  # reward computation & 22 reward terms
│   │   │   ├── legged_robot_dynamics.py # SRB dynamics, finite differences, quat utils
│   │   │   ├── legged_robot_observations.py  # observation assembly & noise
│   │   │   ├── legged_robot_sim_setup.py     # sim/terrain/env creation, heights
│   │   │   ├── legged_robot_callbacks.py     # domain rand, torques, resets, curricula
│   │   │   └── base_task.py
│   │   └── go1/
│   │       ├── go1_config.py            # Go1-specific config (force control params)
│   │       └── ...
│   │
│   ├── utils/
│   │   ├── terrain.py
│   │   ├── math.py
│   │   ├── helpers.py
│   │   └── ...
│   │
│   └── scripts/
│       ├── train.py                     # training entry point
│       └── play.py                      # evaluation / visualization
│
└── rsl_rl/                              # ← git submodule (srb branch)
    ├── rsl_rl/
    │   ├── algorithms/
    │   │   └── ppo.py                   # PPO + HJB loss implementation
    │   ├── modules/
    │   │   └── actor_critic.py
    │   └── storage/
    │       └── rollout_storage.py       # stores srb_dynamics alongside transitions
    └── setup.py
```

### Environment Module Breakdown

The monolithic environment file is decomposed into focused modules using a **mixin pattern**. `LeggedRobot` inherits from all five mixins plus `BaseTask`, so `from .legged_robot import LeggedRobot` remains the single import for all downstream code.

| File | Responsibility |
|------|---------------|
| `legged_robot.py` | Core loop: `__init__`, `step`, `post_physics_step`, `reset_idx`, `_init_buffers` |
| `legged_robot_rewards.py` | `compute_reward`, `_prepare_reward_function`, all `_reward_*` terms |
| `legged_robot_dynamics.py` | `compute_srb_dynamics`, `compute_finite_differences`, `quaternion_to_matrix` |
| `legged_robot_observations.py` | `compute_observations`, `_get_noise_scale_vec` |
| `legged_robot_sim_setup.py` | `create_sim`, terrain creation, `_create_envs`, height queries, debug vis |
| `legged_robot_callbacks.py` | Domain randomization, `_compute_torques`, resets, curricula |

---

## Installation

### Prerequisites

- Ubuntu 20.04 or 22.04
- Python 3.8+
- NVIDIA GPU with CUDA 11.4+ and compatible drivers
- Conda (recommended)

### Step 1 — Install Isaac Gym

Download Isaac Gym Preview 4 from [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym).

```bash
# Extract the downloaded archive
tar -xzf IsaacGym_Preview_4.tar.gz
cd isaacgym/python

# Create and activate a conda environment
conda create -n srb_hjb python=3.8 -y
conda activate srb_hjb

# Install Isaac Gym
pip install -e .

# Verify installation
python -c "import isaacgym; print('Isaac Gym installed successfully')"
```

### Step 2 — Clone This Repository and Load Submodules

```bash
git clone --recurse-submodules https://github.com/YiruWu/legged_gym.git
cd legged_gym
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Step 3 — Install the RL Library (rsl_rl) from the SRB Branch

The PPO + HJB implementation lives on the `srb` branch of the `rsl_rl` submodule. You must check out that branch before installing:

```bash
cd rsl_rl
git checkout srb
pip install -e .
cd ..
```

### Step 4 — Install This Package

```bash
pip install -e .
```

---

## Usage

### Training

```bash
python legged_gym/scripts/train.py --task=go1 --headless
```

Common training flags:

| Flag | Description |
|------|-------------|
| `--task` | Environment name (e.g., `go1`) |
| `--headless` | Run without GUI (recommended for training) |
| `--num_envs` | Number of parallel environments (default: 4096) |
| `--max_iterations` | Total training iterations |
| `--resume` | Resume from latest checkpoint |
| `--checkpoint` | Path to a specific checkpoint to load |

### Evaluation

```bash
python legged_gym/scripts/play.py --task=go1 --checkpoint=<path_to_model.pt>
```

This launches the Isaac Gym viewer and runs the trained policy.

### Configuration

All environment and training hyperparameters are defined in config dataclasses. For the Go1 robot:

- **Environment config:** `legged_gym/envs/go1/go1_config.py`
- **Base config:** `legged_gym/envs/base/legged_robot_config.py`

Key config fields for this project:

```python
class env:
    enable_srb_dynamics = True   # toggle SRB computation in the env

class control:
    control_type = "F"           # "P" = position, "F" = force (GRF)
    action_scale = 1.0
    grf_bias = [0.0, 0.0, 3.27] # per-foot GRF bias (gravity compensation)
```

PPO hyperparameters (passed to the `PPO` class in `rsl_rl`):

```python
hjb_coef = 0.1      # HJB loss weight (0.0 = pure PPO)
gamma = 0.998        # discount factor (ρ = -ln(γ) ≈ 0.002)
learning_rate = 1e-3
num_learning_epochs = 5
num_mini_batches = 4
```

---

## Toggling SRB-HJB

A single config flag controls whether the framework runs as standard PPO or PPO + SRB-HJB:

| Setting | Env Side | Algorithm Side | Behavior |
|---------|----------|---------------|----------|
| `enable_srb_dynamics = True`, `hjb_coef > 0` | Computes SRB dynamics each step | Adds HJB residual to critic loss | **Full PPO + SRB-HJB** |
| `enable_srb_dynamics = False`, `hjb_coef = 0` | Skips SRB (buffer stays zero) | Pure value regression loss | **Standard PPO** |

To switch between the two modes, set both values in your config:

```python
# --- PPO + SRB-HJB mode ---
class env:
    enable_srb_dynamics = True

class algorithm:
    hjb_coef = 0.1

# --- Pure PPO mode ---
class env:
    enable_srb_dynamics = False

class algorithm:
    hjb_coef = 0.0
```

---

## Acknowledgements

This project builds upon the following open-source work:

- [legged_gym](https://github.com/leggedrobotics/legged_gym) — NVIDIA & ETH Zurich (Nikita Rudin), BSD-3-Clause
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — ETH Zurich, Robotic Systems Lab, BSD-3-Clause
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) — NVIDIA Corporation

All modifications and the current version: Copyright © 2025 Yiru Wu, University of Wisconsin-Madison.

## License

This project is released under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.