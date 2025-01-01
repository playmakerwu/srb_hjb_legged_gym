from legged_gym.envs.go1.flat.go1_flat_roller_x_config import Go1FlatRollerXCfg, Go1FlatRollerXCfgPPO

class Go1FlatRollerXPlayCfg( Go1FlatRollerXCfg ):
    class viewer( Go1FlatRollerXCfg.viewer ):
        ref_env = 0
        # 50 robots
        pos = [20, 9, 1]  # [m]
        lookat = [10., 9, 0.]  # [m]

        # single robot
        # pos = [0., 1., 0.5]  # [m]
        # lookat = [0., 0., 0.]  # [m]
    
    class commands( Go1FlatRollerXCfg.commands ):
        class ranges( Go1FlatRollerXCfg.commands.ranges ):
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            # # heading = [-3.14, 3.14]

            lin_vel_x = [1., 1.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]

    # class env( Go1FlatRollerXCfg.env ):
    #     num_envs = 1

    class domain_rand( Go1FlatRollerXCfg.domain_rand ):
        push_robots = False

    class asset( Go1FlatRollerXCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/roller/x/go1_roller_x_obj.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/roller/x/go1_roller_x_simplified_stl.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_model/urdf/roller/p/go1_roller_p_obj.urdf'
        flip_visual_attachments = False

class Go1FlatRollerXPlayCfgPPO( Go1FlatRollerXCfgPPO ):
    class runner ( Go1FlatRollerXCfgPPO.runner):
        experiment_name = 'go1_flat_roller_x'
        # experiment_name = 'go1_flat_roller_cfg_search'
        run_name = ''
        # load_run = -1
        # max_iterations = 300
