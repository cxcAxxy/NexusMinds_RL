import numpy as np
from ....core import Robot
from ..sim.pygym import Gym
import torch

class Franka(Robot):
    def __init__(self,sim:Gym ,cfg):
        # 那么这个地方按照sim，就是以文档里面的 官方文档的prepare_sim为界限
        self.num_actions=cfg.num_actions
        self.num_obs=cfg.num_obs
        self.num_envs=cfg.num_envs
        self.sim=sim
        self.cfg=cfg
        #准备资产，创建环境，为后续的控制做好准备
        self.sim.pre_simulate(cfg.num_envs,cfg.asset,cfg.robot_files,cfg.base_pose,cfg.base_orn)

    def step(self, action) -> None:
        action = action.clone()  # ensure action don't change
        action = torch.clamp(action, self.cfg.action_low, self.cfg.action_high)
        if self.cfg.control_type == "ee":
            ee_displacement = action[:,:3]

            # limit maxium change in position
            ee_displacement = ee_displacement * 0.05  # limit maximum change in position

            # 计算对应的des_pos和des_orn
            des_pos=self.sim.get_ee_position()+ee_displacement
            des_orn=self.sim.get_ee_orientation()
            u=self.sim.ee_pos_to_torque(des_pos,des_orn)

            return u
            # for i in  range(self.cfg.control_decimation):
            #     self.sim.step(u,self.cfg.control_type_sim)

        else:
            raise Exception("需要更新其他的控制方式")
    
    def get_obs(self) -> torch.Tensor:
        # end-effector position and velocity
        ee_position = self.sim.get_ee_position()
        ee_velocity = self.sim.get_ee_velocity()
        
        # fingers opening
        if not self.cfg.block_gripper:
            fingers_width = self.sim.get_fingers_width()
            # 确保 fingers_width 是2维张量 (num_envs, 1)
            if fingers_width.dim() == 1:
                fingers_width = fingers_width.unsqueeze(1)
            observation = torch.cat([ee_position, ee_velocity, fingers_width], dim=1)
        else:
            observation = torch.cat([ee_position, ee_velocity], dim=1)
        return observation

    def reset_ids(self, env_ids):
        # 重置关节位置和速度
        self.sim.reset_joint_states(env_ids)

    def reset(self) -> None:
        """Reset the robot and return the observation."""
        # 重置所有环境
        env_ids = torch.arange(self.num_envs, device=self.sim.device if hasattr(self.sim, 'device') else 'cpu')
        self.reset_ids(env_ids)

    #后面是根据机器的模型，自己定义的一些函数，服务于set_action,get_obs。
    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.sim.get_ee_position()
        target_ee_position = ee_position + ee_displacement


        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles

        target_arm_angles = self.sim.inverse_kinematics(
            link=self.sim.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles


    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.sim.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def check_ee_collision(self, force_threshold = 0.01):
        collision_info = self.sim.get_ee_collision_info()
        collision = collision_info['force_magnitudes'] > force_threshold
        return {'collision_occurred': collision}