import math
import time
from typing import Any, Dict, Iterator, Optional
import torch
import pybullet as p
import pybullet_data
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
import numpy as np
import select
import sys
import cv2
import glfw

# neutral_joint_values,
# render_mode: str = "human",
# n_substeps: int = 1,
# model_path: str = "/home/sxy/桌面/NexusMInds_RL-main/env/assets/xml/franka_emika_panda/scene.xml",

class Mujoco:
    def __init__(
            self,
            cfg
    ):
        self.neutral_joint_values = cfg.neutral_joint_values
        self.render_mode = cfg.render_mode
        self.n_substeps = cfg.n_substeps
        

        try:
            self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {cfg.model_path}: {e}")
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 1.0 / 500
        self.timestep = self.model.opt.timestep
        self.body_id = self.model.body('hand').id
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._init_renderer()
        self.reset()

    def _init_renderer(self):

        """初始化渲染器"""
        self.viewer = None
        self.scene = None

        if self.render_mode == "human":
            # 交互式可视化模式
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"Warning: Could not launch interactive viewer: {e}")
                self.render_mode = "rgb_array"

        elif self.render_mode == "rgb_array":
            # 离屏渲染模式，用于生成图像数组
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()

            # 初始化相机和场景
            mujoco.mjv_defaultCamera(self.cam)
            mujoco.mjv_defaultOption(self.opt)

            # 创建离屏渲染上下文
            self.gl_context = mujoco.GLContext(800, 600)
            self.gl_context.make_current()

            self.con = mujoco.MjrContext(
                self.model,
                mujoco.mjtFontScale.mjFONTSCALE_150.value
            )
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

        else:
            raise ValueError("render_mode must be either 'human' or 'rgb_array'")

    def mjstep(self):
        mujoco.mj_step(self.model, self.data, self.n_substeps)
    
    def step(self, u, control_type="effort"):
        """简化版step函数"""
        # 验证控制类型
        if control_type not in ["effort", "velocity", "position"]:
            raise ValueError("控制类型必须是 'effort', 'velocity' 或 'position'")
        
        u = np.asarray(u)
        
        # 设置控制信号
        if control_type == "effort":
            # # 直接力矩控制
            self.set_joint_angles(u)

            # n_ctrl = min(len(u), self.model.nu)
            # self.data.ctrl[:n_ctrl] = u[:n_ctrl]
        
        elif control_type == "velocity":
            # 简化速度控制
            for i in range(min(len(u), self.model.nu)):
                vel_error = u[i] - self.data.qvel[i]
                self.data.ctrl[i] = 100 * vel_error - 10 * self.data.qvel[i]
        
        elif control_type == "position":
            # 简化位置控制
            for i in range(min(len(u), self.model.nu)):
                pos_error = u[i] - self.data.qpos[i]
                vel_error = -self.data.qvel[i]  # 阻尼项
                self.data.ctrl[i] = 500 * pos_error + 50 * vel_error
        
        # 执行仿真步骤
        mujoco.mj_step(self.model, self.data)
        
        # 渲染
        if self.render_mode == "human" and self.viewer is not None:
            try:
                self.viewer.sync()
            except:
                pass
        
        # return self.data.qpos.copy(), self.data.qvel.copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def close(self):
        if self.viewer:
            self.viewer.close()
        if hasattr(self, 'gl_context'):
            self.gl_context.free()

    #### 要求 ####
    def get_fingers_width(self):
        finger1_pos = self.get_link_position("left_finger")
        finger2_pos = self.get_link_position("right_finger")
        width = np.linalg.norm(finger1_pos - finger2_pos)
        return torch.tensor(np.array(width, dtype=np.float32).reshape(1, -1),
                            dtype=torch.float32,
                            device='cpu')

    def get_ee_velocity(self):
        ee_velocity = self.get_link_velocity("hand")
        return torch.tensor(np.array(ee_velocity, dtype=np.float32).reshape(1, -1),
                            dtype=torch.float32,
                            device='cpu')

    def get_ee_position(self):
        ee_pos = self.data.body(self.body_id).xpos
        return torch.tensor(np.array(ee_pos, dtype=np.float32).reshape(1, -1),
                            dtype=torch.float32,
                            device='cpu')
    
    def get_ee_orientation(self):
        ee_orn = self.data.body(self.body_id).xquat  # wxyz
        return ee_orn
    
    def inverse_kinematics(self, link: str, position: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
        """计算给定末端执行器位置和方向的逆运动学解。

        Args:
            link (str): 末端执行器的链接名称。
            position (np.ndarray): 目标位置 (x, y, z)。
            orientation (np.ndarray): 目标方向四元数 (w, x, y, z) (可选)。"""

    # 完成检验
    def get_joint_angle(self, joint_id: int) -> float:
        """Get the angle of the specified joint in the body.

        Args:
            body (str): Body unique name.
            joint_id (int): Joint id in the body.

        Returns:
            float: The joint angle in radians.
        """

        # 从data.qpos中获取关节位置
        # 需要根据关节的qpos地址索引来定位
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # 检查关节类型并返回相应的角度值
        joint_type = self.model.jnt_type[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            # 铰链关节：返回单个角度值
            return float(self.data.qpos[qpos_adr])
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            # 滑动关节：返回平移量
            return float(self.data.qpos[qpos_adr])
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            # 球关节：返回四元数，需要转换为欧拉角或直接返回四元数
            # 这里返回第一个欧拉角作为示例，实际应用可能需要更复杂的处理
            return float(self.data.qpos[qpos_adr])
        else:
            raise NotImplementedError(f"Joint type {joint_type} not supported")

    def set_joint_neutral(self):
        self.set_joint_angles(self.neutral_joint_values)

    def get_joint_angles(self):
        angles = []
        for i in range(0, 6):
            angle = self.get_joint_angle(i)
            angles.append(angle)
        return np.array(angles)

    def set_joint_angles(self, target_angles):
        for i in range(len(target_angles)):
            joint_id = i 
            angle = target_angles[i]
            self.set_joint_angle(joint_id, angle)

    # 完成检验
    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray = None) -> None:
        """直接通过qpos设置body的位置和方向

        Args:
            body (str): Body名称
            position (np.ndarray): 目标位置 [x, y, z]
            orientation (np.ndarray): 目标方向四元数 [w, x, y, z] (可选)
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id == -1:
            raise ValueError(f"Body '{body}' not found")
        # 从data.xpos中获取body的位置
        # data.xpos是一个(nbody x 3)的数组，每个body对应一行(x, y, z)坐标
        self.data.xpos[body_id] = position
        self.data.xquat[body_id] = orientation if orientation is not None else np.array([1, 0, 0, 0])

        mujoco.mj_forward(self.model, self.data)

    ####要求####

    # 完成检验
    def render(
            self,
            width: int = 720,
            height: int = 480,
            target_position: Optional[np.ndarray] = None,
            distance: float = 1.4,
            yaw: float = 45,
            pitch: float = -30,
            roll: float = 0,
    ):

        if self.render_mode != "rgb_array":
            return None

        target = target_position if target_position is not None else np.zeros(3)

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)

        # self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        # self.cam = mujoco.MjvCamera()
        # self.opt = mujoco.MjvOption()

        cam.lookat = target
        cam.distance = distance
        cam.azimuth = yaw  # 方位角，类似yaw
        cam.elevation = pitch  # 仰角，类似pitch

        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        corrected_rgb = np.flipud(rgb_array)
        depth_array = np.zeros((height, width, 1), dtype=np.float32)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        mujoco.mjr_render(viewport, self.scene, self.con)
        mujoco.mjr_readPixels(rgb_array, depth_array, viewport, self.con)

        return corrected_rgb

    # 完成检验
    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """

        # 获取body的ID
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id == -1:
            raise ValueError(f"Body '{body}' not found")
        # 从data.xpos中获取body的位置
        # data.xpos是一个(nbody x 3)的数组，每个body对应一行(x, y, z)坐标
        position = self.data.xpos[body_id]
        return position.copy()  # 返回副本避免修改原始数据

    # 完成检验
    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as quaternion (w, x, y, z).
        """
        # 获取body的ID
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id == -1:
            raise ValueError(f"Body '{body}' not found in the model")

        # MuJoCo使用w,x,y,z格式的四元数，存储在data.xquat中
        # 注意：MuJoCo的四元数格式是(w, x, y, z)，与某些库的(x, y, z, w)不同
        quaternion = self.data.xquat[body_id]
        return quaternion.copy()

    # 完成检验
    def get_base_rotation(self, body: str, type: str = "euler") -> np.ndarray:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.
            type (str): Type of angle, either "euler" or "quaternion"

        Returns:
            np.ndarray: The rotation.
        """
        if type == "quaternion":
            return self.get_base_orientation(body)
        elif type == "euler":
            # 将四元数转换为欧拉角
            quaternion = self.get_base_orientation(body)
            euler_angles = self._quat_to_euler(quaternion)
            return euler_angles
        else:
            raise ValueError("type must be 'euler' or 'quaternion'.")

    # 不好检验
    def get_base_velocity(self, body: str) -> np.ndarray:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id == -1:
            raise ValueError(f"Body '{body}' not found in the model")

        # 线速度存储在data.cvel中，格式为(线性速度x3, 角速度x3)
        linear_velocity = self.data.cvel[body_id][:3]
        return linear_velocity.copy()

    # 不好检验
    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id == -1:
            raise ValueError(f"Body '{body}' not found in the model")

        # 角速度存储在data.cvel中，格式为(线性速度x3, 角速度x3)
        angular_velocity = self.data.cvel[body_id][3:6]
        return angular_velocity.copy()

    # 完成检验
    def get_link_position(self, link: str) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """

        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        if link_id == -1:
            raise ValueError(f"Body '{link}' not found in the model")

        link_position = self.data.xpos[link_id]
        return link_position.copy()

    # 完成检验
    def get_link_orientation(self, link: str) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as quaternion (w, x, y, z).
        """
        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        if link_id == -1:
            raise ValueError(f"Body '{link}' not found in the model")

        orientation = self.data.xquat[link_id]
        return orientation.copy()

    # 完成检验
    def get_link_velocity(self, link: str) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        # 获取关节ID
        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        if link_id == -1:
            raise ValueError(f"Joint '{link}' not found in body")

        # 从data.cvel中获取关节速度

        velocity_data = self.data.cvel[link_id]
        linear_velocity = velocity_data[0:3]

        return linear_velocity.copy()

    # 完成检验
    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        # 获取关节ID
        link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        if link_id == -1:
            raise ValueError(f"Joint '{link}' not found in body '{body}'")

        # 从data.cvel中获取关节速度

        velocity_data = self.data.cvel[link_id]
        angular_velocity = velocity_data[3:6]

        return angular_velocity.copy()

    # 完成检验
    def get_joint_angle_name(self, joint_name: str) -> float:
        """Get the angle of the specified joint in the body.

        Args:
            body (str): Body unique name.
            joint_name (str): Joint name in the body.

        Returns:
            float: The joint angle in radians.
        """
        # 获取关节ID
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found in body")

        # 从data.qpos中获取关节位置
        # 需要根据关节的qpos地址索引来定位
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # 检查关节类型并返回相应的角度值
        joint_type = self.model.jnt_type[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            # 铰链关节：返回单个角度值
            return float(self.data.qpos[qpos_adr])
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            # 滑动关节：返回平移量
            return float(self.data.qpos[qpos_adr])
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            # 球关节：返回四元数，需要转换为欧拉角或直接返回四元数
            # 这里返回第一个欧拉角作为示例，实际应用可能需要更复杂的处理
            return float(self.data.qpos[qpos_adr])
        else:
            raise NotImplementedError(f"Joint type {joint_type} not supported")

    # 完成检验
    def get_joint_angle_all(self) -> list:
        """获取模型中所有关节的完整信息，包括名称、类型和角度

        Args:
            return_degrees (bool): 是否返回角度值（默认返回弧度）

        Returns:
            list: 包含所有关节信息的字典列表，每个字典包含：
                - joint_name: 关节名称
                - joint_type: 关节类型（字符串表示）
                - joint_type_id: 关节类型ID
                - angle_radians: 关节角度（弧度）
                - angle_degrees: 关节角度（角度）
                - joint_id: 关节ID
                - qpos_address: qpos数组中的地址
        """
        joints_info = []

        # 遍历模型中的所有关节
        for joint_id in range(self.model.njnt):
            # 获取关节名称
            joint_name_id = self.model.name_jntadr[joint_id]
            joint_name = self.model.names[joint_name_id:].split(b'\x00')[0].decode()

            # 获取关节类型
            joint_type_id = self.model.jnt_type[joint_id]
            joint_type_str = self._get_joint_type_string(joint_type_id)

            # 获取qpos地址
            qpos_adr = self.model.jnt_qposadr[joint_id]

            # 根据关节类型获取角度值
            angle_rad = 0.0
            if joint_type_id == mujoco.mjtJoint.mjJNT_HINGE:
                # 铰链关节：单个角度值
                angle_rad = float(self.data.qpos[qpos_adr])
            elif joint_type_id == mujoco.mjtJoint.mjJNT_SLIDE:
                # 滑动关节：平移量
                angle_rad = float(self.data.qpos[qpos_adr])
            elif joint_type_id == mujoco.mjtJoint.mjJNT_BALL:
                # 球关节：四元数的第一个分量
                angle_rad = float(self.data.qpos[qpos_adr])
            elif joint_type_id == mujoco.mjtJoint.mjJNT_FREE:
                # 自由关节：位置的第一个分量
                angle_rad = float(self.data.qpos[qpos_adr])
            else:
                angle_rad = 0.0

            # 计算角度值
            angle_deg = np.degrees(angle_rad)

            joint_info = {
                'joint_name': joint_name,
                'joint_id': joint_id,
                'joint_type': joint_type_str,
                'joint_type_id': joint_type_id,
                'angle_radians': angle_rad,
                'angle_degrees': angle_deg,
            }
            # 'qpos_address': qpos_adr

            joints_info.append(joint_info)

        return joints_info

        # 完成检验

    def get_joint_velocity(self, body: str, joint_name: str) -> float:
        """Get the velocity of the specified joint in the body.

        Args:
            body (str): Body unique name.
            joint_name (str): Joint name in the body.

        Returns:
            float: The joint velocity.
        """
        # 获取关节ID
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found in body '{body}'")

        # 从data.qvel中获取关节速度
        # 需要根据关节的dof地址索引来定位
        dof_adr = self.model.jnt_dofadr[joint_id]

        joint_type = self.model.jnt_type[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            # 铰链关节：返回角速度
            return float(self.data.qvel[dof_adr])
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            # 滑动关节：返回线速度
            return float(self.data.qvel[dof_adr])
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            # 球关节：返回角速度向量的第一个分量
            return float(self.data.qvel[dof_adr])
        else:
            raise NotImplementedError(f"Joint type {joint_type} not supported")

    # 完成检验
    def my_set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of multiple joints using position controllers.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint names.
            angles (np.ndarray): List of target angles.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    # 完成检验
    def set_joint_angle(self, joint: int, angle: float) -> None:
        """Set the angle of a specific joint using its controller.

        Args:
            body (str): Body unique name.
            joint (str): Joint name.
            angle (float): Target angle.
        """
        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        # print(f"Setting joint '{joint_name}'")
        # 查找与关节同名的控制器
        actuator_id = self._find_actuator_for_joint(joint_name)
        if actuator_id is not None:
            # 通过控制器设置目标位置
            self.data.ctrl[actuator_id] = angle
        else:
            # 备用方案：如果没有找到对应控制器，使用默认方法
            print(f"Warning: No controller found for joint '{joint}', using direct qpos setting")
            qpos_adr = self.model.jnt_qposadr[joint]
            self.data.qpos[qpos_adr] = angle

    # 不好检验
    def control_joints(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        """Control the joints using position control with force limits.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint names.
            target_angles (np.ndarray): List of target angles.
            forces (np.ndarray): Maximum forces to apply.
        """
        for joint, angle, force in zip(joints, target_angles, forces):
            actuator_id = self._find_actuator_for_joint(joint)
            if actuator_id is not None:
                # 设置目标位置
                self.data.ctrl[actuator_id] = angle
                # 如果需要动态调整力限制，可以通过修改模型参数实现
                self.model.actuator_gear[actuator_id, 0] = force

    # 辅助方法
    def _find_actuator_for_joint(self, joint_name: str) -> int:
        """查找与关节同名的控制器ID。

        Args:
            joint_name (str): 关节名称

        Returns:
            int: 控制器ID，如果未找到返回None
        """
        for act_id in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            if act_name == joint_name:
                return act_id
        return None

    def quat_to_euler(self, quaternion: np.ndarray) -> np.ndarray:
        """将四元数转换为欧拉角（roll, pitch, yaw）。

        Args:
            quaternion (np.ndarray): 四元数 (w, x, y, z)

        Returns:
            np.ndarray: 欧拉角 (roll, pitch, yaw)
        """
        w, x, y, z = quaternion
        # 转换为欧拉角的实现
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])

    def _euler_to_quat(self, euler_angles: np.ndarray) -> np.ndarray:
        """将欧拉角转换为四元数（辅助函数）

        Args:
            euler_angles: 欧拉角 (roll, pitch, yaw) 弧度制

        Returns:
            np.ndarray: 四元数 (w, x, y, z)
        """
        roll, pitch, yaw = euler_angles

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def _quat_to_euler(self, quaternion: np.ndarray) -> np.ndarray:
        """将四元数转换为欧拉角（roll, pitch, yaw）

        Args:
            quaternion: 四元数 (w, x, y, z)

        Returns:
            np.ndarray: 欧拉角 (roll, pitch, yaw) 弧度制
        """
        w, x, y, z = quaternion

        # 四元数到欧拉角转换公式
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # 使用90度如果超出范围
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _get_joint_type_string(self, joint_type_id: int) -> str:
        """将关节类型ID转换为可读的字符串"""
        joint_type_map = {
            mujoco.mjtJoint.mjJNT_HINGE: "hinge",
            mujoco.mjtJoint.mjJNT_SLIDE: "slide",
            mujoco.mjtJoint.mjJNT_BALL: "ball",
            mujoco.mjtJoint.mjJNT_FREE: "free"
        }
        return joint_type_map.get(joint_type_id, "unknown")

    def refresh(self):
        """刷新MuJoCo仿真状态数据，相当于Isaac Gym中的各种refresh函数"""
        # 执行前向动力学计算，确保所有状态都是最新的
        mujoco.mj_forward(self.model, self.data)
        
        # 刷新刚体状态（相当于refresh_rigid_body_state_tensor）
        # 在MuJoCo中，刚体状态可以通过data.xpos, data.xquat等直接访问
        
        # 刷新关节状态（相当于refresh_dof_state_tensor）
        # qpos和qvel已经在前向动力学计算中更新
        
        # 刷新雅可比矩阵（相当于refresh_jacobian_tensors）
        # 在需要时动态计算，不预先存储
        
        # 刷新质量矩阵（相当于refresh_mass_matrix_tensors）
        # MuJoCo中质量矩阵可以通过mj_fullM函数计算
        if hasattr(self, 'M'):
            mujoco.mj_fullM(self.model, self.M, self.data.qM)

    def ee_pos_to_torque(self, pos_des: np.ndarray, orn_des: np.ndarray ):
        u = self.solve_ik_6d(pos_des, orn_des,
                                n_iter=200,
                                eps_pos=1e-4,
                                eps_rot=1e-3,
                                weight_rot=0.2,
                                max_dq=0.1)
        u = np.array(u, dtype=np.float64).reshape(-1)
        return u

    def orientation_error(self, desired, current):
        """计算四元数姿态误差"""
        # 确保四元数归一化
        desired = desired / np.linalg.norm(desired)
        current = current / np.linalg.norm(current)
        
        # 计算误差四元数: q_error = desired * current_conjugate
        w1, x1, y1, z1 = current
        w2, x2, y2, z2 = desired
        
        q_error = np.array([
            w1*w2 + x1*x2 + y1*y2 + z1*z2,  # 实部
            w1*x2 - x1*w2 - y1*z2 + z1*y2,
            w1*y2 + x1*z2 - y1*w2 - z1*x2, 
            w1*z2 - x1*y2 + y1*x2 - z1*w2
        ])
        
        # 转换为轴角表示（近似小角度误差）
        angle = 2 * np.arccos(np.clip(abs(q_error[0]), 0, 1))
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = q_error[1:] / np.sin(angle/2)
        return axis * angle

    def apply_torque_control(self, pos_des, orn_des=None):
        """应用力矩控制到仿真中"""
        # 计算所需的关节力矩
        torque = self.ee_pos_to_torque(pos_des, orn_des)
        
        # 应用力矩到仿真
        self.data.ctrl[:] = torque
        
        # 执行一步仿真
        mujoco.mj_step(self.model, self.data)
        
        return torque    

    def reset_joint_states(self, env_ids):
        """重置多个环境中机器人的关节状态到初始位置"""
        self.set_joint_neutral()
    
    def solve_ik_6d(self, pos_target, quat_target,
                    n_iter=500, eps_pos=1e-4, eps_rot=1e-3,
                    weight_rot=0.2, max_dq=0.1):
        """
        改进 6D IK：
        - 求解结束后自动恢复机械臂初始状态（data.qpos）
        """
        print(pos_target)
        print(quat_target)
        # ----------- 0. 保存当前机械臂状态 -----------
        original_qpos = self.data.qpos.copy()

        # 1. 处理 pos_des（确保是 np.array shape=(3,)）
        # ------------------------------
        pos_des = pos_target
        orn_des = quat_target
        # ------------------------------
        if isinstance(pos_des, torch.Tensor):
            pos_des = pos_des.detach().cpu().numpy()
        pos_des = np.array(pos_des, dtype=np.float64).reshape(-1)

        # 必须是 3 维
        if pos_des.size != 3:
            raise ValueError(f"pos_des must be 3 elements, got shape {pos_des.shape}")

        # ==========================================================
        # 2. orn_des 变成 numpy (4,)
        # ==========================================================
        if isinstance(orn_des, torch.Tensor):
            orn_des = orn_des.detach().cpu().numpy()
        orn_des = np.array(orn_des, dtype=np.float64).reshape(-1)

        if orn_des.size != 4:
            # 给默认四元数
            orn_des = np.array([0, 0, 0, 1], dtype=np.float64)

        if np.linalg.norm(orn_des) < 1e-6:
            orn_des = np.array([0, 0, 0, 1], dtype=np.float64)

        orn_des = orn_des / np.linalg.norm(orn_des)

        # ==========================================================
        # 后续使用 pos_des 和 orn_des
        # ==========================================================
        quat_target = orn_des  # xyzw

        # 初始化
        if hasattr(self.model, 'key_qpos') and isinstance(self.model.key_qpos, np.ndarray):
            qpos = self.model.key_qpos.copy()
        else:
            qpos = np.zeros(self.model.nq)

        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        for _ in range(n_iter):
            # 1. 雅可比
            J_pos = np.zeros((3, self.model.nv))
            J_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, self.body_id)

            # 2. 当前位姿
            current_pos = np.array(self.data.body(self.body_id).xpos)
            current_quat = np.array(self.data.body(self.body_id).xquat)

            R_current = R.from_quat(current_quat[[1,2,3,0]])
            R_target = R.from_quat(quat_target[[1,2,3,0]])
            R_err_mat = (R_target * R_current.inv()).as_matrix()
            rot_err = 0.5 * np.array([
                R_err_mat[2,1]-R_err_mat[1,2],
                R_err_mat[0,2]-R_err_mat[2,0],
                R_err_mat[1,0]-R_err_mat[0,1]
            ])

            err_pos = pos_target - current_pos
            err_rot = rot_err

            err_pos = err_pos.reshape(3,)
            err_rot = err_rot.reshape(3,)

            err = np.concatenate([err_pos, weight_rot*err_rot])

            if np.linalg.norm(err_pos) < eps_pos and np.linalg.norm(err_rot) < eps_rot:
                break

            # 3. 拼接雅可比
            J6 = np.vstack([J_pos, weight_rot * J_rot])

            dq = np.linalg.pinv(J6) @ err
            dq = np.clip(dq, -max_dq, max_dq)

            qpos[:self.model.nv] += dq

            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)

        # ----------- 4. 计算结束后恢复初始姿态 -----------
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)

        # 返回 IK 解，而不是机械臂状态
        return qpos.copy()
    
    def evaluate_ik_accuracy(self, n_samples=30):
        """
        随机生成末端位姿，求逆解，再做 FK 测误差
        返回平均位置误差（米）与平均姿态误差（度）
        """

        pos_errors = []
        rot_errors = []

        # ----------- 机械臂可达范围（你可根据需要更改） -----------
        # 根据 Franka Panda 的典型可达范围设置
        R_MIN = 0.3     # 最小可达半径
        R_MAX = 0.8     # 最大可达半径
        Z_MIN = 0.1     # 最低 z
        Z_MAX = 0.7     # 最高 z

        for i in range(n_samples):

            # ========== 1. 生成随机有效末端位置 ==========
            # 随机半径、角度生成平面点
            r = np.random.uniform(R_MIN, R_MAX)
            theta = np.random.uniform(-np.pi, np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(Z_MIN, Z_MAX)
            pos_target = np.array([x, y, z])

            # ========== 2. 随机姿态：统一球取样 ==========
            # 随机生成单位四元数 xyzw
            rand = np.random.normal(size=4)
            rand /= np.linalg.norm(rand)
            quat_target = rand.copy()
            # Mujoco 是 wxyz，转换
            quat_target = quat_target[[3,0,1,2]]

            # ========== 3. 求 IK ==========
            qpos_sol = self.solve_ik_6d(pos_target, quat_target)

            # ========== 4. FK，计算实际末端位姿 ==========
            self.data.qpos[:] = qpos_sol
            mujoco.mj_forward(self.model, self.data)

            pos_actual = np.array(self.data.body(self.body_id).xpos)
            quat_actual = np.array(self.data.body(self.body_id).xquat)  # wxyz

            # ========== 5. 误差 ==========

            # ---- 位置误差 ----
            pos_err = np.linalg.norm(pos_actual - pos_target)
            pos_errors.append(pos_err)

            # ---- 姿态误差（使用旋转角度）----
            R_act = R.from_quat(quat_actual[[1,2,3,0]])    # xyzw
            R_tar = R.from_quat(quat_target[[1,2,3,0]])
            R_err = R_act.inv() * R_tar
            rot_err_angle = R_err.magnitude() * (180/np.pi)
            rot_errors.append(rot_err_angle)

            print(f"[{i+1}/{n_samples}] pos_err = {pos_err:.5f} m, "
                f"rot_err = {rot_err_angle:.3f} deg")

        # ========== 6. 计算平均误差 ==========
        avg_pos_err = np.mean(pos_errors)
        avg_rot_err = np.mean(rot_errors)

        print("\n================ IK 评估结果 ================")
        print(f"平均位置误差: {avg_pos_err:.6f} m")
        print(f"平均姿态误差: {avg_rot_err:.3f} deg")
        print("============================================\n")

        return avg_pos_err, avg_rot_err



