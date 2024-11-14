import numpy as np
import os
from datetime import datetime

import isaacgym
from rsl_rl.env import VecEnv
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, update_cfg_from_args, set_seed
import torch
from rsl_rl.runners import OnPolicyRunner
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from typing import Tuple, Any


def get_common_cfgs(args) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO, VecEnv, Any]:
    """ 
    Get the common env_cfg, train_cfg, task_class, sim_params that all searching configs are based on
    Modifying this cfg gives you the cfgs to search and train on
    """
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    task_class = task_registry.get_task_class(name=args.task)

    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    return env_cfg, train_cfg, task_class, sim_params

def train(env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO, task_class, sim_params, args):
    """
    Once you finish modifying the cfgs, say, through a for loop, you can use this function to train for the cfg
    """
    
    # set seed
    set_seed(env_cfg.seed)
    
    # make env
    env = task_class(   cfg=env_cfg,
                        sim_params=sim_params,
                        physics_engine=args.physics_engine,
                        sim_device=args.sim_device,
                        headless=args.headless)
    
    # make alg runner
    train_cfg_dict = class_to_dict(train_cfg)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)  
    ppo_runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
    
    # train
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

