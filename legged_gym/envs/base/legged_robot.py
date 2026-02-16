# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Original code: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Original code: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
#
# All modifications and the current version of this codebase:
# Copyright (c) 2025 Yiru Wu, University of Wisconsin-Madison
# All rights reserved. Yiru Wu has full credit for the current version.

"""
LeggedRobot — Isaac Gym legged-locomotion environment.

This file contains the core simulation loop (init, step, reset, buffers).
Domain-specific logic is split into mixins for clarity:

    legged_robot_rewards.py      – reward computation & individual terms
    legged_robot_dynamics.py     – SRB dynamics, finite differences, quaternion utils
    legged_robot_observations.py – observation assembly & noise scaling
    legged_robot_sim_setup.py    – sim/terrain/env creation, height queries, debug vis
    legged_robot_callbacks.py    – domain-rand callbacks, torque computation,
                                   resets, command resampling, curricula

All external imports (e.g. ``from .legged_robot import LeggedRobot``) remain
unchanged because this file still exposes LeggedRobot at module level.
"""

import numpy as np
import torch
from torch import Tensor

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

# ----- mixin imports -----
from .legged_robot_rewards import RewardsMixin
from .legged_robot_dynamics import DynamicsMixin
from .legged_robot_observations import ObservationsMixin
from .legged_robot_sim_setup import SimSetupMixin
from .legged_robot_callbacks import CallbacksMixin


class LeggedRobot(
    CallbacksMixin,
    SimSetupMixin,
    ObservationsMixin,
    DynamicsMixin,
    RewardsMixin,
    BaseTask,
):
    """Isaac Gym environment for legged robot locomotion training.

    Inherits domain-specific logic from five mixin classes and the
    core task interface from ``BaseTask``.  The MRO ensures that
    mixin methods are found before BaseTask defaults.
    """

    # ==================================================================
    #  Initialisation
    # ==================================================================

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine,
                 sim_device, headless):
        """Parse config, create sim/terrain/envs, init buffers & rewards.

        Args:
            cfg: Environment config object.
            sim_params (gymapi.SimParams): Simulation parameters.
            physics_engine (gymapi.SimType): Must be PhysX.
            sim_device (str): 'cuda' or 'cpu'.
            headless (bool): Run without rendering if True.
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine,
                         sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.t = 0

    # ==================================================================
    #  Step
    # ==================================================================

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step().

        Args:
            actions (Tensor): Shape (num_envs, num_actions_per_env).

        Returns:
            obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras,
            finite_difference
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(
                self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            if self.cfg.control.control_type == "F":
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_jacobian_tensors(self.sim)
        self.post_physics_step()

        # clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        if self.finite_difference is None:
            self.finite_difference = torch.zeros_like(self.obs_buf)

        self.t += 1

        return (self.obs_buf, self.privileged_obs_buf, self.rew_buf,
                self.reset_buf, self.extras, self.finite_difference)

    # ==================================================================
    #  Post-physics step
    # ==================================================================

    def post_physics_step(self):
        """Check terminations, compute observations and rewards.

        Calls ``_post_physics_step_callback()`` for common computations
        and ``_draw_debug_vis()`` if needed.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute rewards, resets
        self.check_termination()
        self.compute_reward()

        # start next round – all states & commands are current for
        # compute_observations(); self.actions is already the last action
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_finite_differences()

        # SRB dynamics: only compute when enabled via config
        if self.enable_srb_dynamics:
            self.compute_srb_dynamics()

        # FIX: zero out augmentation data for just-reset envs.
        # For finite_difference: last_obs was from pre-reset state, so
        #   (post_reset_obs - pre_crash_obs) / dt is a meaningless spike.
        # For srb_dynamics: contact forces are stale (no physics step ran
        #   after the reset teleport), so the dynamics are inconsistent.
        if len(env_ids) > 0:
            self.finite_difference[env_ids] = 0.
            if self.enable_srb_dynamics:
                self.srb_dynamics_buf[env_ids] = 0.

        # in some cases a simulation step might be required to refresh
        # some obs (for example body positions)

        # update last_actions: used for computing action_rate penalty
        # NOT for obsrv – it's the last_last_action for obsrv, but
        # the last action for self.actions
        self.last_actions[:] = self.actions[:]
        self.last_obs = self.obs_buf.clone()
        # update last_dof_vel for computing dof_acc reward
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # ==================================================================
    #  Termination check
    # ==================================================================

    def check_termination(self):
        """Check if environments need to be reset."""
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[
                    :, self.termination_contact_indices, :],
                dim=-1) > 1.,
            dim=1)

        if self.enable_termination_by_height:
            self.termination_by_height_buf = torch.any(
                self.rigid_body_states[
                    :, self.termination_by_height_indices, 2]
                < self.termination_by_height_min_heights,
                dim=1)
            self.reset_buf |= self.termination_by_height_buf

        # no terminal reward for time-outs
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length)
        self.reset_buf |= self.time_out_buf

    # ==================================================================
    #  Environment reset
    # ==================================================================

    def reset_idx(self, env_ids):
        """Reset selected environments.

        Calls ``_reset_dofs``, ``_reset_root_states``,
        ``_resample_commands``, and optionally updates curricula.
        Logs episode info and zeros out buffers.
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if (self.cfg.commands.curriculum
                and (self.common_step_counter
                     % self.max_episode_length == 0)):
            self.update_command_curriculum(env_ids)

        # reset the past actions
        self.actions[env_ids] = 0.

        # reset current robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = (
                torch.mean(self.episode_sums[key][env_ids])
                / self.max_episode_length_s)
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = (
                self.command_ranges["lin_vel_x"][1])
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    # ==================================================================
    #  Buffer initialisation
    # ==================================================================

    def _init_buffers(self):
        """Initialise torch tensors for simulation states and
        processed quantities.
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.control.control_type == "F":
            jacobians = self.gym.acquire_jacobian_tensor(
                self.sim, self.cfg.asset.name)
            self.gym.refresh_jacobian_tensors(self.sim)

        # create wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(
            rigid_body_states).view(self.num_envs, -1, 13)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(
            net_contact_forces).view(self.num_envs, -1, 3)

        if self.cfg.control.control_type == "F":
            self.jacobians = gymtorch.wrap_tensor(jacobians)

        # ---- initialise derived data ----
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx),
            device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch(
            [1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_dofs, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(
            self.num_dofs, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(
            self.num_dofs, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float,
            device=self.device, requires_grad=False)
        if self.cfg.control.control_type in ["P", "V", "T"]:
            self.actions_scaled_including_passive = torch.zeros(
                self.num_envs, self.num_dof, dtype=torch.float,
                device=self.device, requires_grad=False)

        # identify active (actuated) DOFs
        self.active_dof_indices = []
        for idx, dof_name in enumerate(self.dof_names):
            is_passive = False
            for keyword in self.cfg.control.passive_dof_name_keywords:
                if keyword in dof_name:
                    is_passive = True
                    break
            if not is_passive:
                self.active_dof_indices.append(idx)
        if len(self.active_dof_indices) != self.num_actions:
            raise ValueError(
                "Number of active dofs must be equal to the "
                "number of actions")

        # last_actions: used for action_rate penalty, NOT for obsrv
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands,
            dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel,
             self.obs_scales.ang_vel],
            device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0],
            dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float,
            device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                if self.cfg.control.control_type in ["P", "V"]:
                    raise ValueError(
                        f"PD gain of joint {name} were not defined")
            else:
                # check passive joint P gains = 0
                for keyword in self.cfg.control.passive_dof_name_keywords:
                    if (keyword in name) and (self.p_gains[i] != 0.):
                        raise ValueError(
                            f"Passive joint {name} set to have "
                            "non-zero stiffness")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.last_obs = None

        # ground reaction force bias (only for "F" control)
        if self.cfg.control.control_type == "F":
            self.grf_bias = torch.tensor(
                self.cfg.control.grf_bias, dtype=torch.float,
                device=self.device, requires_grad=False)

        # SRB dynamics buffer — always allocated (stays zeros when disabled)
        # so downstream code can safely read self.srb_dynamics_buf regardless
        # of self.enable_srb_dynamics.
        self.srb_dynamics_buf = torch.zeros(
            self.num_envs, 9, dtype=torch.float, device=self.device)

    # ==================================================================
    #  Config parsing
    # ==================================================================

    def _parse_cfg(self, cfg):
        """Extract frequently used values from the config object."""
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt)

        # SRB dynamics toggle (used as critic loss target).
        # Set cfg.env.enable_srb_dynamics = True  to compute SRB each step.
        # Set cfg.env.enable_srb_dynamics = False to skip it entirely.
        # Defaults to False so existing configs without the flag are safe.
        self.enable_srb_dynamics = getattr(
            self.cfg.env, 'enable_srb_dynamics', False)