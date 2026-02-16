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
# Dynamics mixin for LeggedRobot.
# ------------------------------------------------------------------
# Contains:
#   - compute_finite_differences() : numerical obs derivatives
#   - compute_srb_dynamics()       : single-rigid-body dynamics model
#   - _get_feet_world_states()     : foot position / velocity queries
#   - quaternion_to_matrix()       : batched quat -> rotation matrix
# ------------------------------------------------------------------

import torch


class DynamicsMixin:
    """SRB dynamics and finite-difference utilities for LeggedRobot."""

    # -------- finite differences --------

    def compute_finite_differences(self):
        """Compute observation time derivatives via finite differences."""
        if self.last_obs is None:
            self.finite_difference = torch.zeros_like(self.obs_buf)
            return self.finite_difference

        dt = self.dt
        self.finite_difference = (self.obs_buf - self.last_obs) / dt
        return self.finite_difference

    # -------- single-rigid-body dynamics --------

    def compute_srb_dynamics(self):
        """Compute the single-rigid-body (SRB) dynamics:
        base_lin_vel_dot, base_ang_vel_dot, projected_gravity_dot.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                scaled base_lin_vel_dot, scaled base_ang_vel_dot,
                projected_gravity_dot.
        """
        base_lin_vel = self.base_lin_vel.clone()
        base_ang_vel = self.base_ang_vel.clone()
        base_pos = self.root_states[:, :3]
        projected_gravity = self.projected_gravity.clone()
        num_envs = self.obs_buf.shape[0]

        # ---- mass / inertia parameters ----
        m_motor = 0.5
        total_weight = 13.1  # total weight of the robot

        I_base = torch.tensor([
            [0.018144, -0.000247, -0.000291],
            [-0.000247,  0.067993, -0.000042],
            [-0.000291, -0.000042,  0.077422]
        ], dtype=torch.float32, device=self.device)
        diff_x, diff_y = 17.78, 7.62
        I_base[0, 0] += 8 * m_motor * (diff_y / 100) ** 2
        I_base[1, 1] += 8 * m_motor * (diff_x / 100) ** 2
        I_batch = I_base.unsqueeze(0).repeat(num_envs, 1, 1)

        # ---- base linear acceleration ----
        assert base_lin_vel.shape == (num_envs, 3), \
            "Base linear velocity shape mismatch"
        self.gym.refresh_net_contact_force_tensor(self.sim)
        contact_force_w = self.contact_forces[:, self.feet_indices, :].clone()
        Rwb = self.quaternion_to_matrix(self.base_quat)
        Rwb_T = Rwb.transpose(1, 2)
        Rbw_exp = Rwb_T.unsqueeze(1).repeat(1, 4, 1, 1)
        contact_force = torch.matmul(
            Rbw_exp, contact_force_w.unsqueeze(-1)).squeeze(-1)

        # NOTE: contact_indicator removed — it was always ones_like
        # (no-op multiply).  PhysX already reports near-zero forces for
        # swing-phase feet, so the raw contact_force is sufficient.
        base_lin_vel_dot = -torch.cross(base_ang_vel, base_lin_vel, dim=1)
        total_contact_force = torch.sum(contact_force, dim=1)
        base_lin_vel_dot += total_contact_force / total_weight
        base_lin_vel_dot += 9.81 * projected_gravity

        # ---- base angular acceleration ----
        inertia = I_batch
        quat = self.base_quat
        Rwb = self.quaternion_to_matrix(quat)
        Rwb_T = torch.transpose(Rwb, 1, 2)
        Rwb_T_expanded = Rwb_T.unsqueeze(1).repeat(1, 4, 1, 1)
        foot_pos, _ = self._get_feet_world_states()
        base_pos_expanded = base_pos.unsqueeze(1).repeat(1, 4, 1)
        relative_foot_pos = (foot_pos - base_pos_expanded).unsqueeze(-1)
        r_i_B = torch.matmul(
            Rwb_T_expanded, relative_foot_pos).squeeze(-1)
        f_i_B = contact_force  # raw contact forces (see note above)

        tau = torch.sum(
            torch.cross(r_i_B, f_i_B, dim=2), dim=1).unsqueeze(-1)
        inertia_inv = torch.linalg.inv(inertia)
        base_ang_vel_dot = torch.matmul(inertia_inv, tau)

        # ---- projected gravity derivative ----
        projected_gravity_dot = -torch.cross(
            base_ang_vel, projected_gravity, dim=1)

        # ---- fill srb_dynamics buffer ----
        self.srb_dynamics_buf[:, :3] = (
            base_lin_vel_dot * self.obs_scales.lin_vel)
        self.srb_dynamics_buf[:, 3:6] = (
            base_ang_vel_dot.squeeze(-1) * self.obs_scales.ang_vel)
        self.srb_dynamics_buf[:, 6:9] = projected_gravity_dot

        # NOTE: finite_difference and srb_dynamics are kept as separate
        # buffers.  Do NOT overwrite finite_difference[:, 6:9] here —
        # mixing two signals with different noise/scale characteristics
        # corrupts the training data.

        return (base_lin_vel_dot * self.obs_scales.lin_vel,
                base_ang_vel_dot.squeeze(-1) * self.obs_scales.ang_vel,
                projected_gravity_dot)

    # -------- helpers --------

    def _get_feet_world_states(self):
        """Return world-frame foot positions and linear velocities.

        Returns:
            Tuple[Tensor, Tensor]: (foot_pos_w, foot_vel_w)
                each of shape (num_envs, num_feet, 3).
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        foot_pos_w = self.rigid_body_states[:, self.feet_indices, :3]
        foot_vel_w = self.rigid_body_states[:, self.feet_indices, 7:10]
        return foot_pos_w, foot_vel_w

    @staticmethod
    def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
        """Convert a batch of quaternions to rotation matrices.

        Parameters
        ----------
        q : torch.Tensor
            Shape (..., 4).  Component order is (x, y, z, w).

        Returns
        -------
        R : torch.Tensor
            Shape (..., 3, 3).  Rotation matrix that rotates body -> world.
        """
        x, y, z, w = q.unbind(-1)

        xx, yy, zz = x * x, y * y, z * z
        ww = w * w
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = torch.empty(*q.shape[:-1], 3, 3, dtype=q.dtype, device=q.device)

        R[..., 0, 0] = ww + xx - yy - zz
        R[..., 0, 1] = 2.0 * (xy - wz)
        R[..., 0, 2] = 2.0 * (xz + wy)

        R[..., 1, 0] = 2.0 * (xy + wz)
        R[..., 1, 1] = ww - xx + yy - zz
        R[..., 1, 2] = 2.0 * (yz - wx)

        R[..., 2, 0] = 2.0 * (xz - wy)
        R[..., 2, 1] = 2.0 * (yz + wx)
        R[..., 2, 2] = ww - xx - yy + zz

        return R