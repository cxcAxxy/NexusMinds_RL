# config/franka_reach_cfg.py
import numpy as np
from isaacgym import gymutil
from .base_config import BaseConfig
import torch

args = gymutil.parse_arguments(
        description="test Gym Simulation",
        custom_parameters=[
            {"name": "--use_gpu", "type": bool, "default": True, "help": "Use GPU for physics"},
            {"name": "--use_gpu_pipeline", "type": bool, "default": True, "help": "Use GPU pipeline"},
            {"name": "--headless", "type": bool, "default": False, "help": "Run simulation without viewer"},
        ]
    )

class GlobalCfg:
    """全局共享配置 - 统一管理所有组件的共同参数"""
    def __init__(self):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 环境数量 
        self.num_envs = 10

    


class GymCfg:
    """仿真器配置"""
    def __init__(self, args=None):
        # 默认值
        self.headless = False
        self.use_gpu = True
        self.use_gpu_pipeline = True

        # 如果传了 args，就覆盖默认值
        if args is not None:
            for key, value in vars(args).items():
                setattr(self, key, value)



class RobotCfg:
    """机械臂配置"""
    def __init__(self,global_cfg):
        # 控制相关参数
        self.control_type = "ee"      
        self.block_gripper = True 
        self.num_actions = 3            
        self.num_obs = 9
        self.num_envs = global_cfg.num_envs # 修改其他配置一致
        self.control_type_sim = "effort"             

        
        # 模型路径与姿态
        self.asset = "/home/gu/NexusMInds_RL/env/assets"
        self.robot_files = "urdf/franka_description/robots/franka_panda.urdf"
        # 每个机器人的初始位置是一样的吗
        self.base_pose = [0,0,0]  # 每个环境的机器人位置
        self.base_orn = [0,0,0,1] # 每个环境的机器人姿态

        self.ee_link = "panda_hand"
        self.headless = "False"
        self.control_decimation = 6
        self.action_low = -1
        self.action_high = 1



class TaskCfg:
    """Franka Reach 任务配置"""
    def __init__(self,global_cfg):
        self.name = "Reach"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = global_cfg.num_envs  # 修改其他配置一致

        self.reward_type = "dense"
        self.distance_threshold = 0.05

        self.goal_range = 1
        self.get_ee_position = None


class AllCfg:
    """环境总体配置"""
    def __init__(self,global_cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = global_cfg.num_envs
        self.num_achieved_goal = 3
        self.num_desired_goal = 3
        self.max_episode_length = 200
        self.max_episode_length_s = 4.0  # 秒数形式（用于日志统计）
        self.decimation = 4  
        self.control_type_sim = "effort"  


class FrankaReachCfg:
    """总配置类"""
    def __init__(self):
        self.global_cfg = GlobalCfg()
        self.gymcfg = GymCfg(args)
        self.robotcfg = RobotCfg(self.global_cfg)
        self.taskcfg = TaskCfg(self.global_cfg)
        self.all = AllCfg(self.global_cfg)


class LeggedRobotCfgDDPG(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = [516, 256, 256,128]
        critic_hidden_dims = [516, 256, 256,128]
        activation = 'relu'
        n_critics = 2

    class algorithm:
        gamma = 0.99
        tau = 0.005
        batch_size = 256
        max_size = 1_000_000
        lr = 1e-3

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        # 更大的每迭代采样步数与总迭代数（约 200k 环境步）
        num_steps_per_env = 100
        max_iterations = 2500
        buffer_size = 1e6
        save_interval = 100
        experiment_name = 'PPO_Panda'
        run_name = ''
        # 新增：每次迭代更新次数与随机探索步数
        updates_per_iter = 10
        start_random_steps = 1000



