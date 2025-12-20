import numpy as np
from ....core import Robot
from ..sim.pygym_DexGrasp import Gym
import torch


# 测试环境，只测试对应的仿真当中的抓取环节。

class LinkerHand06(Robot):
    def __init__(self, sim: Gym, cfg):
        # 那么这个地方按照sim，就是以文档里面的 官方文档的prepare_sim为界限
        self.num_actions = cfg.num_actions
        self.num_obs = cfg.num_obs
        self.num_envs = cfg.num_envs
        self.robot_num_dofs = cfg.robot_num_dofs
        self.sim = sim
        self.cfg = cfg

        self.kp = torch.zeros(self.num_actions, dtype=torch.float, device=self.sim.device, requires_grad=False)
        self.kv = torch.zeros(self.num_actions, dtype=torch.float, device=self.sim.device, requires_grad=False)

        for i in range(self.robot_num_dofs):
            for dof_name in self.cfg.stiffness.keys():
                self.kp[i] = self.cfg.stiffness[dof_name]
                self.kv[i] = self.cfg.damping[dof_name]

        # 准备资产，创建环境，为后续的控制做好准备
        self.sim.pre_simulate(cfg.num_envs, cfg.asset, cfg.robot_files, cfg.base_pose, cfg.base_orn)

    def step(self, action) -> None:
        action = action.clone()  # ensure action don't change
        action = torch.clamp(action, self.cfg.action_low, self.cfg.action_high)
        if self.cfg.control_type == "ee":
            body_displacement = action[:, :7]
            hand_displacement = action[:, 7:]
            body_displacement = body_displacement * 0.05  # 这里的系数需要考虑
            hand_displacement = hand_displacement * 0.05

            body_joint_pos = self.sim.get_joint_pos()[:, :7]
            body_joint_vel = self.sim.get_joint_vel()[:, :7]

            hand_joint_pos = self.sim.get_joint_pos()[:, 7:]
            hand_joint_vel = self.sim.get_joint_vel()[:, 7:]

            body_kp = self.kp[:7]
            body_kv = self.kv[:7]

            hand_kp = self.kp[7:]
            hand_kv = self.kv[7:]

            distance = self.sim.get_hand_to_object_distance()
            distance = torch.norm(distance, dim=-1)
            mask = distance > 0.1

            u1 = self.sim.body_joint_to_torque(body_displacement, body_joint_pos, body_joint_vel, body_kp , body_kv)
            #加判断条件
            
            u2 = self.sim.hand_joint_to_torque(hand_displacement, hand_joint_pos, hand_joint_vel, hand_kp, hand_kv)
            u2[mask] = 0

            u = torch.cat([u1, u2], dim=1)

            return u
            # for i in  range(self.cfg.control_decimation):
            #     self.sim.step(u,self.cfg.control_type_sim)

        else:
            raise Exception("需要更新其他的控制方式")

    def get_obs(self) -> torch.Tensor:
        # end-effector position  velocity orientation  angular velocity
        dof_pos = self.sim.get_joint_pos().squeeze(-1)
        ee_position = self.sim.get_ee_position()
        ee_orientation = self.sim.get_ee_orientation()
        ee_velocity = self.sim.get_ee_velocity()
        ee_angular_velocity = self.sim.get_ee_angular_velocity()
        middle_point_to_object_distance = self.sim.get_hand_to_object_distance()
        middle_point = self.sim.get_fingers_mid_point()
        obj_pos = self.sim.get_obj_position()
        obj_quat = self.sim.get_obj_quaternion()


        observation = torch.cat(
            [dof_pos, ee_position, ee_orientation, ee_velocity,  
             ee_angular_velocity, middle_point_to_object_distance, middle_point, obj_pos, obj_quat]
            , dim=1)
        return observation

    def reset_ids(self, env_ids):
        # 重置关节位置和速度
        self.sim.reset_joint_states(env_ids)
        self.sim.reset_object_states(env_ids)

    def reset(self) -> None:
        """Reset the robot and return the observation."""
        # 重置所有环境
        env_ids = torch.arange(self.num_envs, device=self.sim.device if hasattr(self.sim, 'device') else 'cpu')
        self.reset_ids(env_ids)

    # 后面是根据机器的模型，自己定义的一些函数，服务于set_action,get_obs。
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

    def check_hand_collision(self, force_threshold=0.01):
        collision_info = self.sim.get_ee_collision_info()
        collision = collision_info['force_magnitudes'] > force_threshold
        return {'collision_occurred': collision}
    
    def check_finger_collision(self):
        collision_info = self.sim.get_finger_collision_info()
        collision = collision_info['collision_flags']
        return {'finger_collision_occurred': collision}
    
    def check_body_collision(self):
        collision_info = self.sim.get_body_collision_info()
        collision = collision_info['collision_flags']
        return {'body_collision_occurred': collision}
    
    def check_object_reset(self):
        reset_info = self.sim.get_object_reset_info()
        reset_flags = reset_info['reset_obj']
        return {'obj_reset_occurred': reset_flags}