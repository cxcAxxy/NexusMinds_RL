import os
import numpy as np
from datetime import datetime
import sys

# 首先导入包含 Isaac Gym 的配置和环境模块
from configs.LinkerHandGrasp_config import LinkGraspCfg
from env.TaskRobotEnv import LinkerHandGraspGym

# 然后导入可能包含 PyTorch 的 rsl_rl 模块
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO
from rsl_rl.utils import class_to_dict

def train():

    cfg = LinkGraspCfg()
    train_cfg = class_to_dict(rslCfgPPO())
    
    # 使用正确的环境类和配置
    env = LinkerHandGraspGym(cfg)
    
    # 将 Runner 放在与环境一致的设备上

    ppo_runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=cfg.global_cfg.logdir, device=str(env.device))
    ppo_runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)

if __name__ == '__main__':
    train()

#python train.py --sim_device "cuda:1" --graphics_device_id 1 --logdir logs_c2:3_alpha_pos:0.5