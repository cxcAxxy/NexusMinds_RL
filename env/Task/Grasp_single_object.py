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
        self.alpha_mid = cfg.alpha_mid
        self.alpha_pos = cfg.alpha_pos
        self.alpha_down = cfg.alpha_down
        self.alpha_z = cfg.alpha_z
        self.grasp_goal_distance = cfg.reward_scales["grasp_goal_distance"]
        self.grasp_mid_point = cfg.reward_scales["grasp_mid_point"]
        # self.pos_reach_distance = cfg.reward_scales["pos_reach_distance"]
        self.finger_collision_reset = cfg.reward_scales["finger_collision_reset"]
        self.body_collision_reset = cfg.reward_scales["body_collision_reset"]
        self.obj_reset = cfg.reward_scales["obj_reset"]
        self.hand_down = cfg.reward_scales["hand_down"]
        self.finger_z_distance = cfg.reward_scales["finger_z_distance"]

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
        goals_pos = torch.tensor([0.5, 0, 0.5], dtype=torch.float32, device=self.device)
        goals_pos = goals_pos.unsqueeze(0).expand(self.num_envs, 3)
        self.goal = goals_pos

    # def _sample_goals(self, env_ids: int) -> torch.Tensor:

    #     """为若干环境随机生成目标 (num_envs, 3)"""
    #     # 保证env先重置
    #     goals_pos = torch.tensor([0.5, 0, 0.5], dtype=torch.float32, device=self.device)
    #     self.goals_pos = goals_pos[env_ids]
    #     return self.goals_pos

    def is_success(self) -> torch.Tensor:
        """判断是否成功 (num_envs,)"""

        achieved_goal = self.get_achieved_goal()


        d = torch.norm( achieved_goal- self.goal, dim=-1)
        return d < self.distance_threshold
     
    def reward_grasp_goal_distance(self):
        achieved_goal = self.get_achieved_goal()

        d = self.goal[:, 2] - achieved_goal[:, 2] 
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

    # def reward_pos_reach_distance(self):

    #     hand_base_pos = self.sim.get_hand_base_pos()

    #     d = torch.norm(hand_base_pos - self.sim.get_obj_position(), dim=-1)

    #     reward_pos = torch.exp(-self.alpha_pos * d)

    #     return self.pos_reach_distance * reward_pos

    def reward_hand_down(self):
        x_hand = self.sim.get_rigid_body_x_axis_world()

        world_down = torch.tensor(
            [0.0, 0.0, 1.0],
            device=x_hand.device
        ).expand_as(x_hand)

        cos_sim = torch.sum(x_hand * world_down, dim=1)

        # 按照区间
        # 0.86 按照姿态调整
        # if cos_sim > 0 and cos_sim < 0.86:
        #     reward = 0
        # else:

        cos_sim = torch.clamp(cos_sim, 0.0, 1.0)

        reward = torch.exp(-self.alpha_down * (1.0 - cos_sim))
        return -self.hand_down * reward

    def reward_finger_collision_reset(self):
        reset_events = self.sim.check_reset_events()
        finger_reset = reset_events['finger_collision'].float()

        return -self.finger_collision_reset * finger_reset

    def reward_body_collision_reset(self):
        reset_events = self.sim.check_reset_events()
        body_reset = reset_events['body_collision'].float()

        return -self.body_collision_reset * body_reset

    def reward_obj_reset(self):
        reset_events = self.sim.check_reset_events()
        obj_reset = reset_events['obj_reset'].float()

        return -self.obj_reset * obj_reset
    
    def reward_finger_z_distance(self):
        distance = self.sim.get_finger_z_distance()
        # distance = torch.norm(distance, dim=-1)
        distance = 0.064-distance

        return self.finger_z_distance * distance