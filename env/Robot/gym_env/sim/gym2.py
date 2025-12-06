import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
from pickle import NONE, TRUE
from cvxopt import setseed
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Dict

import numpy as np

import random
import time
import math

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch

import panda_gym.assets

import utils


class Gym:
    def __init__(self, args, headless):

        # 控制器参数验证
        self.controller = args.controller

        # 设置随机数
        set_seed(1, True)
        torch.set_printoptions(precision=10, sci_mode=False)

        # 获取仿真接口
        self.gym = gymapi.acquire_gym()
        self.headless = headless
        self.args = args

        # 控制器参数
        self.damping = 0.05  # IK参数
        self.kp = 150.0  # OSC比例增益
        self.kd = 2.0 * np.sqrt(self.kp)
        self.kp_null = 10.0
        self.kd_null = 2.0 * np.sqrt(self.kp_null)
        # 设置torch设备
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'

        # 配置物理仿真参数
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        if args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 8
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.0
            self.sim_params.physx.contact_offset = 0.001
            self.sim_params.physx.friction_offset_threshold = 0.001
            self.sim_params.physx.friction_correlation_distance = 0.0005
            self.sim_params.physx.num_threads = args.num_threads
            self.sim_params.physx.use_gpu = args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")

        # 初始化关键属性
        self.num_envs = 0
        self.envs = []
        self.obj_idxs = []
        self.obj_handles = []
        self.robot_handles = []
        self.robot_idxs = []
        self.hand_idxs = []

        # create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine,
                                       self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        if self.headless == False:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")

        # 禁用梯度
        torch.set_grad_enabled(False)

    def pre_simulate(self):
        obj_asset = self.create_obj_asset()
        obj_pose = self.set_obj_pose()
        robot_asset = self.create_robot_asset()
        robot_pose = self.set_robot_pose()
        envs = self.create_envs(self.envs)
        return

    def set_camera_pos(self):
        """设置初始相机位置"""
        self.cam_pos = gymapi.Vec3(10, 10, 8)
        self.cam_target = gymapi.Vec3(0, 0, 2)
        self.middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        if self.headless == False:
            self.gym.viewer_camera_look_at(self.viewer, self.middle_env, self.cam_pos, self.cam_target)

    def create_obj_asset(self):
        """创建物体资产"""
        asset_root = ""
        obj_asset_file = ""
        asset_options = gymapi.AssetOptions()
        obj_asset = self.gym.load_asset(self.sim, asset_root, obj_asset_file, asset_options)
        self.obj_asset = obj_asset
        return obj_asset

    def set_obj_pose(self):
        obj_pose = gymapi.Transform()
        # 具体位姿参数

        self.obj_pos = obj_pose
        return obj_pose

    def create_robot_asset(self):
        asset_root = ""
        robot_asset_file = ""
        asset_options = gymapi.AssetOptions()
        # 参数设置

        # 设置摩擦系数
        robot_material_props = gymapi.MaterialProperties()
        robot_material_props.static_friction = 0.8  # 静摩擦系数
        robot_material_props.dynamic_friction = 0.6  # 动摩擦系数
        robot_material_props.restitution = 0.1  # 恢复系数（弹性）

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options, robot_material_props)
        self.robot_asset = robot_asset

        # 配置robot的DOF属性
        self.robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        robot_lower_limits = self.robot_dof_props["lower"]
        robot_upper_limits = self.robot_dof_props["upper"]
        robot_ranges = robot_upper_limits - robot_lower_limits
        robot_mids = 0.5 * (robot_upper_limits + robot_lower_limits)

        # 根据控制器类型设置驱动模式 索引位置修改
        if self.controller == "ik":
            self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)  # 控制方法
            self.robot_dof_props["stiffness"][:7].fill(400.0)  # 刚度修改
            self.robot_dof_props["damping"][:7].fill(40.0)  # 阻尼修改
        else:  # osc
            self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self.robot_dof_props["stiffness"][:7].fill(0.0)
            self.robot_dof_props["damping"][:7].fill(0.0)

        # 夹爪设置
        self.robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.robot_dof_props["stiffness"][7:].fill(800.0)
        self.robot_dof_props["damping"][7:].fill(40.0)

        # 默认DOF位置
        robot_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        self.default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        self.default_dof_pos[:7] = robot_mids[:7]  # 具体索引位置修改
        self.default_dof_pos[7:] = robot_upper_limits[7:]
        return robot_asset

    def set_robot_pose(self):
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0, 0)
        self.robot_pos = robot_pose
        return robot_pose

    def add_ground_plane(self, z=-1):
        """创建空地"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = -z
        self.gym.add_ground(self.sim, plane_params)

    def create_envs(self, num_envs=5):
        """创建并行环境"""
        self.num_envs = num_envs
        self.num_per_row = int(math.sqrt(num_envs))
        self.spacing = 2.0
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)

        print("Creating %d environments" % num_envs)
        self.add_ground_plane()
        self.envs = []
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)

    def create_instances(self):
        """在所有环境中创建物体实例"""
        print("🎯 Creating object instances...")

        self.obj_handles = []
        self.obj_idxs = []
        self.robot_handles = []
        self.robot_idxs = []
        self.hand_idxs = []
        for i, env in enumerate(self.envs):
            # 设置物体位姿
            obj_pose = self.set_obj_pose()

            # 创建桌子
            obj_handle = self.gym.create_actor(
                env, self.obj_asset, obj_pose, f"obj_{i}", i, 0
            )
            self.obj_handles.append(obj_handle)
            # 设置物体颜色

            # 创建robot
            robot_pose = self.set_robot_pose()
            robot_handle = self.gym.create_actor(
                env, self.robot_asset, robot_pose, f"robot_{i}", i, 2
            )
            self.robot_handles.append(robot_handle)

            # 设置DOF属性
            self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

            # 设置初始DOF状态
            default_dof_state = np.zeros(len(self.default_dof_pos), gymapi.DofState.dtype)
            default_dof_state["pos"] = self.default_dof_pos
            self.gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)

            # 设置初始位置目标
            self.gym.set_actor_dof_position_targets(env, robot_handle, self.default_dof_pos)

            # 获取手部索引
            hand_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # 获取物体索引
            obj_idx = self.gym.get_actor_rigid_body_index(env, obj_handle, 0, gymapi.DOMAIN_SIM)
            self.obj_idxs.append(obj_idx)
            robot_idx = self.gym.get_actor_rigid_body_index(env, robot_handle, 0, gymapi.DOMAIN_SIM)
            self.robot_idxs.append(robot_idx)

    def refresh_all_tensors(self):
        """刷新基础张量确保数据可用"""
        self.gym.prepare_sim(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def get_all_tensors(self):

        # 获取刚体状态张量
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # 获取DOF状态张量
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

    def get_dof_pos(self):
        """设置DOF状态张量"""
        # 计算每个环境的DOF数量
        total_dofs = self.dof_states.shape[0]
        dofs_per_env = total_dofs // (2 * self.num_envs)

        # 重新塑形DOF状态
        num_dofs = len(self.default_dof_pos)

        if self.num_envs == 1:
            # 单环境情况
            self.dof_pos = self.dof_states[:, 0].view(1, num_dofs)
            self.dof_vel = self.dof_states[:, 1].view(1, num_dofs)
        else:
            # 多环境情况
            self.dof_pos = self.dof_states[:, 0].view(self.num_envs, num_dofs)
            self.dof_vel = self.dof_states[:, 1].view(self.num_envs, num_dofs)
        return self.dof_pos, self.dof_vel

    def get_base_pos(self):
        self.base_pos = self.dof_states[:1, 0]
        self.base_vel = self.dof_states[:1, 1]
        return self.base_pos, self.base_vel

    def get_gripper_width(self):
        gripper_sep = self.dof_pos[:, 7] + self.dof_pos[:, 8]
        return gripper_sep

    def get_hand_tensors(self):
        if self.hand_idxs:
            if self.num_envs == 1:
                self.hand_pos = self.rb_states[self.hand_idxs[0]:self.hand_idxs[0] + 1, :3]
                self.hand_rot = self.rb_states[self.hand_idxs[0]:self.hand_idxs[0] + 1, 3:7]
                self.hand_vel = self.rb_states[self.hand_idxs[0]:self.hand_idxs[0] + 1, 7:]
            else:
                self.hand_pos = self.rb_states[self.hand_idxs, :3]
                self.hand_rot = self.rb_states[self.hand_idxs, 3:7]
                self.hand_vel = self.rb_states[self.hand_idxs, 7:]
        return self.hand_pos, self.hand_rot, self.hand_vel

    def get_obj_tensors(self):
        if self.obj_idxs:
            if self.num_envs == 1:
                self.obj_pos = self.rb_states[self.obj_idxs[0]:self.obj_idxs[0] + 1, :3]
                self.obj_rot = self.rb_states[self.obj_idxs[0]:self.obj_idxs[0] + 1, 3:7]
                self.obj_vel = self.rb_states[self.obj_idxs[0]:self.obj_idxs[0] + 1, 7:]
            else:
                self.obj_pos = self.rb_states[self.obj_idxs, :3]
                self.obj_rot = self.rb_states[self.obj_idxs, 3:7]
                self.obj_vel = self.rb_states[self.obj_idxs, 7:]
        return self.obj_pos, self.obj_rot, self.obj_vel

    def get_jacobian_tensor(self):
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        j_eef = jacobian[:, :, :, :7]
        self.j_eef = j_eef

    def get_mass_matrix_tensor(self):
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        mm = mm[:, :7, :7]
        self.mm = mm

    def setup_action_tensors(self):
        """初始化动作张量"""

        # 位置动作张量（用于IK控制和夹爪控制）
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)

        # 力矩动作张量（用于OSC控制）
        self.effort_action = torch.zeros_like(self.pos_action)

    def reboot_state(self):
        self.default_dof_pos_tensor = to_torch(self.default_dof_pos, device=self.device)

    def control_ik(self, dpose):
        """IK控制器"""
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def control_osc(self, dpose):
        """OSC控制器"""
        mm_inv = torch.inverse(self.mm)
        m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.hand_vel.unsqueeze(-1))

        # 零空间控制
        j_eef_inv = m_eef @ self.j_eef @ mm_inv
        u_null = self.kd_null * -self.dof_vel[:, :7] + self.kp_null * (
                (self.default_dof_pos_tensor[:7].view(1, -1, 1) - self.dof_pos[:, :7] + np.pi) % (2 * np.pi) - np.pi)
        u_null = self.mm @ u_null
        u += (torch.eye(7, device=self.device).unsqueeze(0) -
              torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null

        return u.squeeze(-1)

    def update_state_variables(self):
        if not hasattr(self, 'rb_states') or not hasattr(self, 'dof_states'):
            return

        # 手部状态
        if self.hand_idxs:
            self.hand_pos = self.rb_states[self.hand_idxs, :3]
            self.hand_rot = self.rb_states[self.hand_idxs, 3:7]
            self.hand_vel = self.rb_states[self.hand_idxs, 7:]

        # 物体状态
        if self.obj_idxs:
            self.obj_pos = self.rb_states[self.obj_idxs, :3]
            self.obj_rot = self.rb_states[self.obj_idxs, 3:7]
            self.obj_vel = self.rb_states[self.obj_idxs, 7:]

        # DOF状态（优化形状处理）
        num_dofs = len(self.default_dof_pos)
        if self.num_envs == 1:
            self.dof_pos = self.dof_states[:, 0].view(1, num_dofs)
            self.dof_vel = self.dof_states[:, 1].view(1, num_dofs)
        else:
            self.dof_pos = self.dof_states[:, 0].view(self.num_envs, num_dofs)
            self.dof_vel = self.dof_states[:, 1].view(self.num_envs, num_dofs)

    def step_simulation(self, num_steps=1000):
        """步进物理仿真并更新所有状态数据"""
        for _ in range(num_steps):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.refresh_all_tensors()
            self.update_state_variables()
            # update viewer
            if self.headless == False:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)

        return True

    def cleanup(self):
        """清理并释放仿真资源"""
        if self.headless == False:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        self.translations = None
        self.quaternions = None


def set_seed(seed, torch_deterministic=False):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed