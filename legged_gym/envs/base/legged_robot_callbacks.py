# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Original code: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Original code: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
#
# All modifications and the current version of this codebase:
# Copyright (c) 2025 Yiru Wu, University of Wisconsin-Madison
# All rights reserved. Yiru Wu has full credit for the current version.
#
# Callbacks mixin for LeggedRobot.
# ------------------------------------------------------------------
# Contains:
#   - _process_rigid_shape_props()   : friction randomisation
#   - _process_dof_props()           : DOF limits & armature
#   - _process_rigid_body_props()    : base mass / CoM randomisation
#   - _post_physics_step_callback()  : resampling, heading, push
#   - _resample_commands()           : random velocity commands
#   - _compute_torques()             : PD / V / T / F controllers
#   - _reset_dofs()                  : reset joint positions & vels
#   - _reset_root_states()           : reset base pose & velocities
#   - _push_robots()                 : random push perturbation
#   - _update_terrain_curriculum()   : game-inspired terrain leveling
#   - update_command_curriculum()    : increasing command range
# ------------------------------------------------------------------

import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *

from legged_gym.utils.math import wrap_to_pi, torch_rand_sqrt_float


class CallbacksMixin:
    """Physics callbacks, torque computation, resets, and curricula."""

    # -------- asset property callbacks (called during env creation) --------

    def _process_rigid_shape_props(self, props, env_id):
        """Randomise friction of each environment.

        Called during environment creation.
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1],
                    (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """Store DOF position / velocity / torque limits from the URDF
        and apply joint armature.

        Called during environment creation.
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float,
                device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float,
                device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float,
                device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0]
                     + self.dof_pos_limits[i, 1]) / 2
                r = (self.dof_pos_limits[i, 1]
                     - self.dof_pos_limits[i, 0])
                self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)
                self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit)

        if self.cfg.domain_rand.randomize_joint_armature:
            rng = self.cfg.domain_rand.added_joint_armature_range
            props["armature"][:] = (
                self.nonrand_joint_armature
                + np.random.uniform(-rng, rng, len(props)))
        else:
            props["armature"][:] = self.nonrand_joint_armature

        return props

    def _process_rigid_body_props(self, props, env_id):
        """Randomise base mass and center-of-mass position.

        Called during environment creation.
        """
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.randomize_base_com_pos:
            rng = self.cfg.domain_rand.added_com_pos_range
            props[0].com.x += np.random.uniform(-rng[0], rng[0])
            props[0].com.y += np.random.uniform(-rng[1], rng[1])
            props[0].com.z += np.random.uniform(-rng[2], rng[2])
        return props

    # -------- post-physics callbacks --------

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards,
        and observations.

        Default behaviour: compute angular velocity command based on
        target and heading, compute measured terrain heights, and
        randomly push robots.
        """
        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if (self.cfg.domain_rand.push_robots
                and (self.common_step_counter
                     % self.cfg.domain_rand.push_interval == 0)):
            self._push_robots()

    # -------- command resampling --------

    def _resample_commands(self, env_ids):
        """Randomly select commands for some environments."""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    # -------- torque computation --------

    def _compute_torques(self, actions):
        """Compute torques from actions.

        Actions can be interpreted as position or velocity targets
        given to a PD controller, or directly as scaled torques.

        NOTE: torques must have the same dimension as the number of
        DOFs, even if some DOFs are not actuated.
        """
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if self.cfg.control.control_type in ["P", "V", "T"]:
            # passive dofs have actions_scaled = 0; their p_gains are
            # checked to be 0 in _init_buffers(), so action values
            # are not used – only d_gains matter for passive dofs
            self.actions_scaled_including_passive[
                :, self.active_dof_indices] = actions_scaled

        if control_type == "P":
            torques = (
                self.p_gains
                * (self.actions_scaled_including_passive
                   + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel)
        elif control_type == "V":
            torques = (
                self.p_gains
                * (self.actions_scaled_including_passive - self.dof_vel)
                - self.d_gains
                * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            torques = self.actions_scaled_including_passive
        elif control_type == "F":
            # "grf" = force from foot to ground
            grf_des_world = quat_rotate(
                self.root_states[:, 3:7].repeat_interleave(
                    len(self.feet_indices), dim=0),
                actions_scaled.view(
                    self.num_envs * len(self.feet_indices), 3)
                + self.grf_bias
            ).view(self.num_envs, len(self.feet_indices), 1, 3)

            torques = torch.sum(
                torch.matmul(
                    grf_des_world,
                    self.jacobians[:, self.feet_indices, :3, 6:]),
                dim=1).view(self.num_envs, self.num_dofs)
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # -------- resets --------

    def _reset_dofs(self, env_ids):
        """Reset DOF positions (randomly around default) and zero
        velocities for selected environments.
        """
        self.dof_pos[env_ids] = (
            self.default_dof_pos
            * torch_rand_float(0.5, 1.5,
                               (len(env_ids), self.num_dof),
                               device=self.device))
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """Reset ROOT state positions and velocities for selected
        environments.  Sets base position based on the curriculum
        and randomises base velocities within [-0.5, 0.5].
        """
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """Random pushes the robots (emulates impulse via base velocity)."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    # -------- curriculum --------

    def _update_terrain_curriculum(self, env_ids):
        """Game-inspired terrain curriculum.

        Robots that walked far enough progress to harder terrains;
        those that walked less than half go to simpler terrains.
        """
        if not self.init_done:
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2],
            dim=1)
        move_up = distance > self.terrain.env_length / 2
        move_down = (
            (distance < torch.norm(self.commands[env_ids, :2], dim=1)
             * self.max_episode_length_s * 0.5)
            * ~move_up)
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(
                self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """Increase command range if tracking reward is high enough."""
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
                / self.max_episode_length
                > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0., self.cfg.commands.max_curriculum)