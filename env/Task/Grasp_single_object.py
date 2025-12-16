from .utils import distance
from typing import Any, Dict, List
import torch
from ..core import Task


class Grasp_single_object(Task):
    def __init__(self, sim, cfg) -> None:
        super().__init__(sim)
        self.sim = sim
        self.reward_type = cfg.reward_type
        self.distance_threshold = cfg.distance_threshold
        self.device = cfg.device
        self.num_envs = cfg.num_envs

        # 参数
        self.alpha_mid =cfg.alpha_mid
        self.alpha_pos= cfg.alpha_pos
        self.c1=cfg.c1
        self.c2=cfg.c2
        self.c3=cfg.c3
        self.c4=cfg.c4
        self.c5 = cfg.c5
        self.c6 = cfg.c6

        self.grasp_goal_distance = cfg.reward_scales["grasp_goal_distance"]
        self.grasp_mid_point = cfg.reward_scales["grasp_mid_point"]
        self.pos_reach_distance = cfg.reward_scales["pos_reach_distance"]

        # 初始化目标缓存 (num_envs, 3)
        self.goal = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def get_obs(self) -> torch.Tensor:

        """返回任务观测，可自行扩展"""
        # 这个地方应该是要返回Object pose（position + quaternion）
        obj_pos_and_quat = torch.cat([self.sim.get_obj_position(), self.sim.get_obj_quaternion()], dim=1)
        return obj_pos_and_quat

    def get_achieved_goal(self) -> torch.Tensor:

        """获得当前物体的"""
        get_achieved_goal= self.sim.get_obj_position()
        return get_achieved_goal

    def reset_ids(self, env_ids: torch.Tensor) -> torch.Tensor:

        """只为指定环境重置目标"""
        goals = self._sample_goals(env_ids)
        self.goal[env_ids] = goals

    def _sample_goals(self, env_ids: int) -> torch.Tensor:

        """为若干环境随机生成目标 (num_envs, 3)"""
        # 保证env先重置
        goals_pos = self.sim.get_obj_position()##
        goals_pos[env_ids ,2] += 0.2 ##
        self.goals_pos = goals_pos[env_ids]
        return self.goals_pos

    def is_success(self) -> torch.Tensor:
        """判断是否成功 (num_envs,)"""

        achieved_goal = self.get_achieved_goal()


        d = torch.norm( achieved_goal- self.goal, dim=-1)
        return d < self.distance_threshold

    def reward_grasp_goal_distance(self):
        achieved_goal = self.get_achieved_goal()

        d = torch.norm(achieved_goal - self.goal, dim=-1)
        if self.reward_type == "sparse":
            goal_distance = (d > self.distance_threshold).float()
        else:
            goal_distance = d

        return self.grasp_goal_distance * (0.2 - goal_distance)

    def reward_grasp_mid_point(self):
        two_fingers_mid = self.sim.get_two_fingers_mid_point()
        d_mid = two_fingers_mid - self.sim.get_obj_position()

        dist = torch.norm(d_mid, dim=-1)  # [N]
        r_neg = torch.exp(-self.alpha_mid * dist)  # exp(-α_neg * d_neg_min)

        return self.grasp_mid_point * r_neg

    def reward_pos_reach_distance(self):

        fingers_mid = self.sim.get_fingers_mid_point()

        d = torch.norm(fingers_mid - self.sim.get_obj_position())

        reward_pos = torch.exp(-self.alpha_pos * d)

        return self.pos_reach_distance * reward_pos

    def reward_test(self):

        return 1