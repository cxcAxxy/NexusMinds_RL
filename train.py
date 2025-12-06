import os
import numpy as np
from datetime import datetime
import sys

# 首先导入包含 Isaac Gym 的配置和环境模块
from configs.Robot_config import FrankaReachCfg
from env.TaskRobotEnv import FrankaReachFixedPointGym

# 然后导入可能包含 PyTorch 的 rsl_rl 模块
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO, envCfg
from rsl_rl.utils import class_to_dict

def train():

    cfg = FrankaReachCfg()
    train_cfg = class_to_dict(rslCfgPPO())
    
    # 使用正确的环境类和配置
    env = FrankaReachFixedPointGym(cfg)
    
    # 将 Runner 放在与环境一致的设备上

    ppo_runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir='logs', device=str(env.device))
    ppo_runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)

if __name__ == '__main__':
    train()
