import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymutil
from env.Robot.gym_env.sim.pygym import Gym
from env.Robot.gym_env.instance.franka import Franka
from env.Task.Reach_fixed_point import Reach
from configs.Robot_config import FrankaReachCfg
import torch

asset_root="/home/ymy/Desktop/NexusMind_rl/env/assets"
urdf_file="urdf/franka_description/robots/franka_panda.urdf"

def test_reach():
    args = gymutil.parse_arguments(
        description="test Gym Simulation",
        custom_parameters=[
            {"name": "--use_gpu", "type": bool, "default": True, "help": "Use GPU for physics"},
            {"name": "--use_gpu_pipeline", "type": bool, "default": True, "help": "Use GPU pipeline"},
            {"name": "--headless", "type": bool, "default": False, "help": "Run simulation without viewer"},
        ]
    )

    cfg = FrankaReachCfg()
    _Gym= Gym(args)
    robot=Franka(_Gym,cfg.robotcfg)
    task = Reach(_Gym, cfg.taskcfg)

    # env_ids = torch.arange(cfg.all.num_envs, device=cfg.all.device)
    # task.reset_ids(env_ids)
    # print("Goals:", task.goal)
    # print("EE positions:", task.get_achieved_goal())

    # desired_goal = torch.rand((cfg.all.num_envs, 3), device=task.device)
    # achieved_goal = task.get_achieved_goal()    
    # reward = task.compute_reward(achieved_goal, desired_goal)

    # print("Achieved Goals:\n", achieved_goal)
    # print("Desired Goals:\n", desired_goal)
    # print("Reward:\n", reward)

    for step in range(1):
        print(f"init EE positions:", task.get_achieved_goal())
        action = torch.rand((cfg.all.num_envs, robot.num_actions), device=cfg.all.device) * 2 - 1
        robot.step(action)

        desired_goal = torch.rand((cfg.all.num_envs, 3), device=task.device)
        achieved_goal = task.get_achieved_goal()    
        reward = task.compute_reward(achieved_goal, desired_goal)

        print("Achieved Goals:\n", achieved_goal)
        print("Desired Goals:\n", desired_goal)
        print("Reward:\n", reward)

        env_ids = torch.arange(cfg.all.num_envs, device=cfg.all.device)
        task.reset_ids(env_ids)
        print("Achieved Goals:\n", achieved_goal)
        print("Desired Goals:\n", desired_goal)


if __name__ == "__main__":
    test_reach()
