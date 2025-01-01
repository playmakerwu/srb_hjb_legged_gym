from legged_gym.envs.go1.flat.go1_flat_config import Go1FlatCfg, Go1FlatCfgPPO

class Go1FlatPlayCfg( Go1FlatCfg ):
    class viewer( Go1FlatCfg.viewer ):
        ref_env = 0
        # 50 robots
        pos = [20, 9, 1]  # [m]
        lookat = [10., 9, 0.]  # [m]

        # single robot
        # pos = [0., -1., 1.]  # [m]
        # lookat = [0., 0., 0.]  # [m]
    
    class commands( Go1FlatCfg.commands ):
        class ranges( Go1FlatCfg.commands.ranges ):
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            # # heading = [-3.14, 3.14]

            lin_vel_x = [1., 1.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]

    # class env( Go1FlatCfg.env ):
    #     num_envs = 1
    
    class domain_rand( Go1FlatCfg.domain_rand ):
        push_robots = False
        # push_interval_s = 1
        # max_push_vel_xy = 1.

    class asset( Go1FlatCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/go1_dae.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/go1_obj.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/go1_simplified_stl.urdf'
        # flip_visual_attachments = True
        flip_visual_attachments = False

class Go1FlatPlayCfgPPO( Go1FlatCfgPPO ):
    class runner ( Go1FlatCfgPPO.runner):
        experiment_name = 'go1_flat'
        run_name = ''
        # load_run = -1
        # max_iterations = 300
