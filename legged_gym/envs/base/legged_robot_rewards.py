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
# Reward function mixin for LeggedRobot.
# ------------------------------------------------------------------
# Contains:
#   - compute_reward()          : main reward aggregation loop
#   - _prepare_reward_function(): builds the list of active reward fns
#   - _reward_*()               : individual reward terms
# ------------------------------------------------------------------

import torch
from isaacgym.torch_utils import quat_rotate_inverse


class RewardsMixin:
    """All reward-related methods for LeggedRobot."""

    # -------- reward aggregation --------

    def compute_reward(self):
        """Compute rewards.
        Calls each reward function which had a non-zero scale
        (processed in self._prepare_reward_function()), adds each term
        to the episode sums and to the total reward.
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, which will be called to
        compute the total reward.  Looks for ``self._reward_<NAME>``
        where ``<NAME>`` are names of all non-zero reward scales in cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float,
                              device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    # -------- individual reward terms --------

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity."""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize base height away from target."""
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        """Penalize torques."""
        return torch.sum(
            torch.square(self.torques[:, self.active_dof_indices]), dim=1)

    def _reward_dof_vel(self):
        """Penalize dof velocities."""
        return torch.sum(
            torch.square(self.dof_vel[:, self.active_dof_indices]), dim=1)

    def _reward_dof_acc(self):
        """Penalize dof accelerations."""
        return torch.sum(torch.square(
            (self.last_dof_vel[:, self.active_dof_indices]
             - self.dof_vel[:, self.active_dof_indices]) / self.dt), dim=1)

    def _reward_action_rate(self):
        """Penalize changes in actions."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_vert_virt_leg(self):
        """Penalize virtual leg's deviation from vertical direction."""
        return torch.sum(torch.square(
            self.rigid_body_states[:, self.feet_indices, :2]
            - self.rigid_body_states[:, self.virtual_leg_upper_vtx_indices, :2]
        ), dim=(1, 2))

    def _reward_collision(self):
        """Penalize collisions on selected bodies."""
        return torch.sum(
            1. * (torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :],
                dim=-1) > 0.1),
            dim=1)

    def _reward_termination(self):
        """Terminal reward / penalty."""
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        """Penalize dof positions too close to the limit."""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        """Penalize dof velocities too close to the limit.
        Clip to max error = 1 rad/s per joint to avoid huge penalties.
        """
        return torch.sum(
            (torch.abs(self.dof_vel)
             - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
             ).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        """Penalize torques too close to the limit."""
        return torch.sum(
            (torch.abs(self.torques)
             - self.torque_limits * self.cfg.rewards.soft_torque_limit
             ).clip(min=0.),
            dim=1)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)."""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_x_exp(self):
        """Tracking of linear velocity commands (x axis)."""
        x_vel_error = torch.square(
            self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-x_vel_error / self.cfg.rewards.tracking_sigma_vx)

    def _reward_tracking_lin_vel_y_exp(self):
        """Tracking of linear velocity commands (y axis)."""
        y_vel_error = torch.square(
            self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-y_vel_error / self.cfg.rewards.tracking_sigma_vy)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)."""
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_z_exp(self):
        """Tracking of angular velocity commands (yaw)."""
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma_wz)

    def _reward_feet_air_time(self):
        """Reward long steps.
        Need to filter the contacts because the contact reporting of
        PhysX is unreliable on meshes.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # reward only on first contact with the ground
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1)
        # no reward for zero command
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _get_minus_c_h_ft(self, z_ft):
        """Helper: ``-c * (z_ft - foot_radius)``."""
        return (-self.cfg.rewards.low_feet_antislip_sigmoid_stiffness
                * (z_ft - self.cfg.rewards.foot_radius))

    def _reward_low_feet_antislip(self):
        """Penalize feet tangential velocity when feet height is low.

        rwd = sum_{feet}{ sigmoid(-c * h_ft) * || v_ft_tangential ||^2 }
        NOTE: "tangential" = v_xy; "h_ft" = z_ft - foot_radius (flat terrain only).
        """
        return torch.sum(
            torch.sigmoid(self._get_minus_c_h_ft(
                self.rigid_body_states[:, self.feet_indices, 2]))
            * self.rigid_body_states[:, self.feet_indices, 7:9]
              .square().sum(dim=2),
            dim=1)

    def _reward_low_roller_y_antislip(self):
        """Penalize roller local-frame y velocity when roller height is low.

        Uses local frame y velocity to approximate the tangential velocity
        in the axial-gnd-aligned direction.
        NOTE: all limitations of ``_reward_low_feet_antislip`` apply.
        """
        return torch.sum(
            torch.sigmoid(self._get_minus_c_h_ft(
                self.rigid_body_states[:, self.feet_indices, 2]))
            * quat_rotate_inverse(
                self.rigid_body_states[:, self.feet_indices, 3:7].view(-1, 4),
                self.rigid_body_states[:, self.feet_indices, 7:10].view(-1, 3)
            )[:, 1].view(self.num_envs, len(self.feet_indices)).square(),
            dim=1)

    def _reward_stumble(self):
        """Penalize feet hitting vertical surfaces."""
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1)

    def _reward_stand_still(self):
        """Penalize motion at zero commands."""
        return (
            torch.sum(torch.abs(
                self.dof_pos[:, self.active_dof_indices]
                - self.default_dof_pos[:, self.active_dof_indices]), dim=1)
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        )

    def _reward_feet_contact_forces(self):
        """Penalize high contact forces."""
        return torch.sum(
            (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
             - self.cfg.rewards.max_contact_force).clip(min=0.),
            dim=1)