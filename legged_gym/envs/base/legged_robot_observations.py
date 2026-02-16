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
# Observation mixin for LeggedRobot.
# ------------------------------------------------------------------
# Contains:
#   - compute_observations() : assemble the observation vector
#   - _get_noise_scale_vec() : per-obs-channel noise scaling
# ------------------------------------------------------------------

import torch


class ObservationsMixin:
    """Observation construction and noise for LeggedRobot."""

    def compute_observations(self):
        """Assemble the observation buffer from current state.

        Layout (without heights):
            [0:3]   base_lin_vel * scale
            [3:6]   base_ang_vel * scale
            [6:9]   projected_gravity
            [9:12]  commands * scale
            [12:24] (dof_pos - default) * scale   (active DOFs only)
            [24:36] dof_vel * scale                (active DOFs only)
            [36:48] last actions
        """
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos)[:, self.active_dof_indices]
            * self.obs_scales.dof_pos,
            self.dof_vel[:, self.active_dof_indices]
            * self.obs_scales.dof_vel,
            self.actions  # this is already the last action
        ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1)
                    - self.cfg.rewards.base_height_target
                    - self.measured_heights,
                    -1, 1.)
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                (2 * torch.rand_like(self.obs_buf) - 1)
                * self.noise_scale_vec
            )

    def _get_noise_scale_vec(self, cfg):
        """Build a vector used to scale the noise added to observations.

        NOTE: Must be adapted when changing the observation structure.
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # multiplied by obs_scales because noise is added to scaled obs
        noise_vec[:3] = (noise_scales.lin_vel * noise_level
                         * self.obs_scales.lin_vel)
        noise_vec[3:6] = (noise_scales.ang_vel * noise_level
                          * self.obs_scales.ang_vel)
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.   # commands
        noise_vec[12:24] = (noise_scales.dof_pos * noise_level
                            * self.obs_scales.dof_pos)
        noise_vec[24:36] = (noise_scales.dof_vel * noise_level
                            * self.obs_scales.dof_vel)
        noise_vec[36:48] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:] = (noise_scales.height_measurements * noise_level
                              * self.obs_scales.height_measurements)
        return noise_vec