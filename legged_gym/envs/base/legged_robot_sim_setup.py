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
# Simulation setup mixin for LeggedRobot.
# ------------------------------------------------------------------
# Contains:
#   - create_sim()            : top-level sim + terrain + envs
#   - set_camera()            : viewer camera positioning
#   - _create_ground_plane()  : flat plane terrain
#   - _create_heightfield()   : heightfield terrain
#   - _create_trimesh()       : triangle-mesh terrain
#   - _create_envs()          : per-env actor creation & indexing
#   - _get_env_origins()      : grid / curriculum env placement
#   - _init_height_points()   : height measurement sample points
#   - _get_heights()          : sample terrain heights
#   - _draw_debug_vis()       : wireframe debug visualisation
# ------------------------------------------------------------------

import os
import numpy as np
import torch

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw


class SimSetupMixin:
    """Simulation creation, terrain, environments, and height queries."""

    # -------- top-level sim creation --------

    def create_sim(self):
        """Create simulation, terrain, and environments."""
        self.up_axis_idx = 2  # 2 for z, 1 for y
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. "
                "Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    # -------- camera --------

    def set_camera(self, position, lookat):
        """Set camera position and direction."""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # -------- terrain creation --------

    def _create_ground_plane(self):
        """Add a ground plane to the simulation."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Add a heightfield terrain to the simulation."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device))

    def _create_trimesh(self):
        """Add a triangle-mesh terrain to the simulation."""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'),
            tm_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device))

    # -------- environment creation --------

    def _create_envs(self):
        """Load the robot asset and create per-environment actors.

        Steps:
            1. Load URDF/MJCF asset.
            2. For each environment: create env, set shape/DOF/body props,
               create actor.
            3. Store body indices for feet, penalised contacts, termination
               contacts, and optional termination-by-height.
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # not joint armature – value added to all links' inertia diagonals
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            robot_asset)

        # body / dof names
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names
                      if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend(
                [s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend(
                [s for s in body_names if name in s])

        # store joint armatures
        self.nonrand_joint_armature = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            for dof_name_keyword in self.cfg.asset.joint_armature.keys():
                if dof_name_keyword in self.dof_names[i]:
                    self.nonrand_joint_armature[i] = (
                        self.cfg.asset.joint_armature[dof_name_keyword])
                    break

        base_init_state_list = (
            self.cfg.init_state.pos + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(
                -1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose,
                self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(
                env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # optional: virtual-leg upper vertex indices for vert_virt_leg reward
        if 'vert_virt_leg' in self.reward_scales.keys():
            if self.reward_scales['vert_virt_leg'] != 0:
                vtx_names = [s for s in body_names
                             if self.cfg.asset.virt_leg_upper_vtx_name in s]
                if len(vtx_names) == 0:
                    raise ValueError(
                        "No virtual leg upper vtx found when trying to "
                        "include vert_virt_leg reward")
                self.virtual_leg_upper_vtx_indices = []
                for i in range(len(vtx_names)):
                    self.virtual_leg_upper_vtx_indices.append(
                        self.gym.find_actor_rigid_body_handle(
                            self.envs[0], self.actor_handles[0], vtx_names[i]))

        # feet indices
        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device,
            requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i])

        # penalised contact indices
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long,
            device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = (
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0],
                    penalized_contact_names[i]))

        # termination contact indices
        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long,
            device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = (
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0],
                    termination_contact_names[i]))

        # optional termination by low height
        self.enable_termination_by_height = False
        if self.cfg.asset.terminate_by_low_height.keys():
            self.enable_termination_by_height = True
            if not self.cfg.terrain.mesh_type == "plane":
                raise ValueError(
                    "cfg.asset.termination_by_height is only supported "
                    "on cfg.terrain.mesh_type='plane'")
            n_bodies = len(self.cfg.asset.terminate_by_low_height.keys())
            self.termination_by_height_indices = torch.zeros(
                n_bodies, dtype=torch.long, device=self.device,
                requires_grad=False)
            self.termination_by_height_min_heights = torch.zeros(
                (self.num_envs, n_bodies), dtype=torch.float,
                device=self.device, requires_grad=False)
            for idx, (name, h_min) in enumerate(
                    self.cfg.asset.terminate_by_low_height.items()):
                self.termination_by_height_indices[idx] = (
                    self.gym.find_actor_rigid_body_handle(
                        self.envs[0], self.actor_handles[0], name))
                self.termination_by_height_min_heights[:, idx] = h_min

    # -------- environment origins --------

    def _get_env_origins(self):
        """Set environment origins.
        On rough terrain the origins are defined by the terrain platforms,
        otherwise a regular grid is created.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device).to(torch.float))
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(
                torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    # -------- height measurement --------

    def _init_height_points(self):
        """Return points at which the height measurements are sampled
        (in base frame).

        Returns:
            Tensor: shape (num_envs, num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y,
            device=self.device, requires_grad=False)
        x = torch.tensor(
            self.cfg.terrain.measured_points_x,
            device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs, self.num_height_points, 3,
            device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Sample terrain heights at required points around each robot.

        Points are offset by the base's position and rotated by the
        base's yaw.
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(
                self.num_envs, self.num_height_points,
                device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = (
                quat_apply_yaw(
                    self.base_quat[env_ids].repeat(1, self.num_height_points),
                    self.height_points[env_ids])
                + (self.root_states[env_ids, :3]).unsqueeze(1))
        else:
            points = (
                quat_apply_yaw(
                    self.base_quat.repeat(1, self.num_height_points),
                    self.height_points)
                + (self.root_states[:, :3]).unsqueeze(1))

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return (heights.view(self.num_envs, -1)
                * self.terrain.cfg.vertical_scale)

    # -------- debug visualisation --------

    def _draw_debug_vis(self):
        """Draw wireframe height-measurement points (slow)."""
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(
                self.base_quat[i].repeat(heights.shape[0]),
                self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(
                    gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer,
                    self.envs[i], sphere_pose)