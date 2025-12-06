from .utils import distance
from typing import Any, Dict
import torch
from ..core import Task

class Reach_fixed_point(Task):
    def __init__(self, sim, cfg) -> None:
        super().__init__(sim)
        self.sim = sim
        self.reward_type = cfg.reward_type
        self.distance_threshold = cfg.distance_threshold
        self.device = cfg.device
        self.num_envs = cfg.num_envs

        # 获取末端执行器位置的函数
        self.get_ee_position = sim.get_ee_position()

        get_ball_positions = sim.get_ball_positions()
        self.goal = get_ball_positions.clone()
        

        # # 目标范围
        # self.goal_range_low = torch.tensor([-cfg.goal_range / 2, -cfg.goal_range / 2, 0.0],
        #                                    dtype=torch.float32, device=self.device)
        # self.goal_range_high = torch.tensor([cfg.goal_range / 2, cfg.goal_range / 2, cfg.goal_range],
        #                                     dtype=torch.float32, device=self.device)

        # # 初始化目标缓存 (num_envs, 3)
        # self.goal = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def get_obs(self) -> torch.Tensor:
        """返回任务观测，可自行扩展"""
        return self.goal

    def get_achieved_goal(self) -> torch.Tensor:
        """获取所有环境的末端执行器位置 (num_envs, 3)"""
        ee_pos = self.sim.get_ee_position()
        return ee_pos


    def reset_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """只为指定环境重置目标"""
        # goals = self._sample_goals(len(env_ids))
        # self.goal[env_ids] = goals

        # 固定目标位姿
        # self.goal[env_ids] = torch.tensor([0.5, 0.3, 0.25], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        # # 通过设置的 goal 和 没有旋转的 orn
        # pos = torch.tensor([0.3, 0.2, 0.1], device=self.device)  # 固定位置
        # orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # 固定姿态
        # self.sim.set_actor_pose("target", pos, orn,env_ids)

    
        ball_positions = self.sim.get_ball_positions()  
        goals = ball_positions[env_ids]       
        
        self.goal[env_ids] = goals
        


    # def _sample_goals_rand(self, num_envs: int) -> torch.Tensor:
    #     """为若干环境随机生成目标 (num_envs, 3)"""
    #     rand = torch.rand((num_envs, 3), device=self.device)
    #     goals = self.goal_range_low + (self.goal_range_high - self.goal_range_low) * rand
    #     return goals

    def is_success(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """判断是否成功 (num_envs,)"""
        d = torch.norm(achieved_goal - desired_goal, dim=-1)
        return d < self.distance_threshold
    
    def check_ee_collision(self, force_threshold = 0.01):
        collision_info = self.sim.get_ee_collision_info()
        collision = collision_info['force_magnitudes'] > force_threshold
        return {'collision_occurred': collision}
    
    def compute_collision_penalty(self):
        """计算固定碰撞惩罚"""
        collision_info = self.check_ee_collision()
        fixed_penalty = -2.0
        penalty = fixed_penalty * collision_info['collision_occurred'].float()
        return penalty

    def compute_reward(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """计算奖励 (num_envs,)"""
        d = torch.norm(achieved_goal - desired_goal, dim=-1)
        if self.reward_type == "sparse":
            task_reward = -(d > self.distance_threshold).float()
        else:
            task_reward = -d
        
        collision_penalty = self.compute_collision_penalty()

        success_bonus = 4.0
        success = self.is_success(achieved_goal, desired_goal)
        success_reward = success_bonus * success.float()
        
        total_reward = task_reward + collision_penalty + success_reward

        return total_reward
