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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.go1.flat.go1_flat_grf_config import Go1FlatGrfCfg, Go1FlatGrfCfgPPO

class Go1FlatGrfPlayCfg( Go1FlatGrfCfg ):
    class viewer( Go1FlatGrfCfg.viewer ):
        ref_env = 0
        # 50 robots
        pos = [20, 9, 1]  # [m]
        lookat = [10., 9, 0.]  # [m]

        # single robot
        # pos = [0., -1., 1.]  # [m]
        # lookat = [0., 0., 0.]  # [m]
    
    class commands( Go1FlatGrfCfg.commands ):
        class ranges( Go1FlatGrfCfg.commands.ranges ):
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            # # heading = [-3.14, 3.14]

            lin_vel_x = [1., 1.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]

    # class env( Go1FlatGrfCfg.env ):
    #     num_envs = 1
    
    class domain_rand( Go1FlatGrfCfg.domain_rand ):
        push_robots = False
        # push_interval_s = 1
        # max_push_vel_xy = 1.

    class asset( Go1FlatGrfCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/go1_dae.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/go1_obj.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/go1_simplified_stl.urdf'
        # flip_visual_attachments = True
        flip_visual_attachments = False


class Go1FlatGrfPlayCfgPPO( Go1FlatGrfCfgPPO ):
    class runner ( Go1FlatGrfCfgPPO.runner):
        experiment_name = 'go1_flat_grf_play'
        run_name = ''
        # load_run = -1
        # max_iterations = 300
