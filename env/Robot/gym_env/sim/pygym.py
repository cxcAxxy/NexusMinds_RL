import random
import time

import numpy as np

# gym应该要实现的接口
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import math

import torch
#后期，配置文件的参数，仿真的一些可视化参数。
from ...utils import *


# 这个args是需要从命令行当中进行一个读取，使用isaac gym的命令行读取
class Gym():
    def __init__(self,args):
        self.args=args
        self.gym=gymapi.acquire_gym()

        # 配置物理仿真参数
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        if args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 4
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.num_threads = args.num_threads
            self.sim_params.physx.use_gpu = args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")

        # 根据参数确定张量设备
        if getattr(args, 'use_gpu', False) or getattr(args, 'use_gpu_pipeline', False):
            compute_id = getattr(args, 'compute_device_id', 0)
            self.device = torch.device(f'cuda:{compute_id}') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        # create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine,self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        if self.args.headless == False:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")

    def create_robot_asset(self,urdf_file,asset_root):
        # 创建模板
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        print("Loading asset '%s' from '%s'" % (urdf_file, asset_root))
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, urdf_file, asset_options)

    def create_table_asset(self):
        # 创建模板
        table_dims = gymapi.Vec3(0.6, 1.0, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    def create_box_asset(self):
        box_size = 0.1
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)

    def create_ball_asset(self):
        radius = 0.025
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        self.ball_asset = self.gym.create_sphere(self.sim, radius, asset_options)


    #后面接入参数，设置pd参数等等
    def set_dof_states_and_propeties(self):

        # set default DOF states
        self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"][:7] = self.robot_mids[:7]

        # set DOF control properties (except grippers)
        self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
        self.robot_dof_props["stiffness"][:7].fill(0.0)
        self.robot_dof_props["damping"][:7].fill(0.0)

        # set DOF control properties for grippers
        self.robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.robot_dof_props["stiffness"][7:].fill(800.0)
        self.robot_dof_props["damping"][7:].fill(40.0)


    def create_envs_and_actors(self,num_envs,base_pos,base_orn):
        # 首先是根据 base_pos和base_orn创建对应的 gyapi.Transform()
        pose=gymapi.Transform()
        pose.p =gymapi.Vec3(base_pos[0],base_pos[1],base_pos[2])
        pose.r=gymapi.Quat(base_orn[0],base_orn[1],base_orn[2],base_orn[3])

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.05)

        box_poses = [
            gymapi.Transform(),
            gymapi.Transform(),
            gymapi.Transform(),
            gymapi.Transform(),
            gymapi.Transform(),
        ]
        box_poses[0].p = gymapi.Vec3(table_pose.p.x + 0.15, table_pose.p.y + 0.2, 0.1 + 0.5 * 0.1)
        box_poses[0].r = gymapi.Quat(0, 0, 2, 1)
        box_poses[1].p = gymapi.Vec3(table_pose.p.x - 0.15, table_pose.p.y + 0.2, 0.1 + 0.5 * 0.1)
        box_poses[1].r = gymapi.Quat(0, 0, 8, 1)
        box_poses[2].p = gymapi.Vec3(table_pose.p.x + 0.15, table_pose.p.y - 0.2, 0.1 + 0.5 * 0.1)
        box_poses[2].r = gymapi.Quat(0, 0, 5, 1)
        box_poses[3].p = gymapi.Vec3(table_pose.p.x - 0.15, table_pose.p.y - 0.2, 0.1 + 0.5 * 0.1)
        box_poses[3].r = gymapi.Quat(0, 0, 10, 1)

     


        self.num_envs=num_envs
        self.envs=[]
        self.table_handles=[]
        self.table_idxs=[]
        self.box_handles=[]
        self.box_idxs=[]
        self.ball_handles=[]
        self.ball_idxs=[]
        self.ee_handles=[]
        self.ee_idxs=[]
        self.left_finger_idxs=[]
        self.right_finger_idxs=[]
        self.init_pos_list=[]
        self.init_orn_list=[]
        
        # 环境对应的参数系数
        self.num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, self.num_per_row)
            self.envs.append(env)

            ball_pose = self.generate_random_ball_pose()

            table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 0)
            self.table_handles.append(table_handle)
            table_idx = self.gym.find_actor_rigid_body_index(env, table_handle, "table", gymapi.DOMAIN_SIM)
            self.table_idxs.append(table_idx)
            
            for j, pose in enumerate(box_poses):
                    box_handle = self.gym.create_actor(env, self.box_asset, pose, f"box_{i}_{j}", i, 0)
                    self.box_handles.append(box_handle)
                    box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
                    self.box_idxs.append(box_idx)

            ball_handle = self.gym.create_actor(env, self.ball_asset, ball_pose, "ball", i, 1)
            red_color = gymapi.Vec3(1.0, 0.0, 0.0)  
            self.gym.set_rigid_body_color(env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, red_color)
            self.ball_handles.append(ball_handle)
            ball_idx = self.gym.get_actor_rigid_body_index(env, ball_handle, 0, gymapi.DOMAIN_SIM)
            self.ball_idxs.append(ball_idx)

            # Add franka
            robot_handle = self.gym.create_actor(env, self.robot_asset, pose, "franka", i, 1)

            # Set initial DOF states
            self.gym.set_actor_dof_states(env, robot_handle, self.default_dof_state, gymapi.STATE_ALL)

            # Set DOF control properties
            self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

            # Get inital ee pose
            ee_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle, "panda_hand")
            self.ee_handles.append(ee_handle)
            ee_pose = self.gym.get_rigid_transform(env, ee_handle)
            self.init_pos_list.append([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z])
            self.init_orn_list.append([ee_pose.r.x, ee_pose.r.y, ee_pose.r.z, ee_pose.r.w])

            left_finger_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "left_finger_joint", gymapi.DOMAIN_SIM)
            self.left_finger_idxs.append(left_finger_idx)

            right_finger_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "right_finger_joint", gymapi.DOMAIN_SIM)
            self.right_finger_idxs.append(right_finger_idx)

            # Get global index of ee in rigid body state tensor
            ee_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.ee_idxs.append(ee_idx)


    def set_camera(self):
        # Point camera at middle env
        if getattr(self.args, 'headless', False):
            return
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def pre_simulate(self,num_envs,asset_root,asset_file,base_pos,base_orn):
        self.create_plane()
        self.create_robot_asset(asset_file,asset_root)
        self.create_table_asset()
        self.create_box_asset()
        self.create_ball_asset()

        # get joint limits and ranges for Franka
        self.robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        robot_lower_limits = self.robot_dof_props['lower']
        robot_upper_limits = self.robot_dof_props['upper']
        robot_ranges = robot_upper_limits - robot_lower_limits
        # 设置一下robot_mids,可能是用来初始化的作用，这个地方稍微记忆一下.
        self.robot_mids = 0.5 * (robot_upper_limits + robot_lower_limits)
        self.robot_num_dofs = len(self.robot_dof_props)

        self.set_dof_states_and_propeties()
        # 创建环境和设置实例
        self.create_envs_and_actors(num_envs,base_pos,base_orn)

        self.set_camera()
        self.gym.prepare_sim(self.sim)
        self.get_state_tensors()

    def get_state_tensors(self):

        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)

        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)

        self._contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(self._contact_forces)


        # 拆分位置与速度分量
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)

        # Jacobian entries for end effector
        self.ee_index = self.gym.get_asset_rigid_body_dict(self.robot_asset)["panda_hand"]
        self.j_eef = self.jacobian[:, self.ee_index - 1, :]

        # Prepare mass matrix tensor
        self._massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(self._massmatrix)
        self.refresh()

    # 仿真步骤步进一次
    def step(self,u,control_type):
        if control_type == "effort" :
            # Set tensor action
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(u))
        elif control_type == "velocity":
            self.gym.set_dof_velocity_target_tensor(self.sim,gymtorch.unwrap_tensor(u))
        elif control_type == "position":
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(u))
        else :
            raise ValueError(f"Unsupported control type: {self.control_type}. Must be one of ['effort', 'velocity', 'position'].")
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.refresh()
    # Step rendering (skip when headless)
        if not getattr(self.args, 'headless', False):
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

    def refresh(self):
        # 看上层从底层读取了什么,那么这个地方就进行了一个什么refresh

        # 末端位姿(root_state_tensor)
        # 关节角、速度(dof_state_tensor)
        # 接触力(net_contact_force_tensor)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def ee_pos_to_torque(self,pos_des,orn_des):
        # 由末端位置控制,由雅可比矩阵等计算出对应的力矩
        kp = 5
        kv = 2 * math.sqrt(kp)
        # 使用 之前，先要同步一下，刚创建的时候可能数据为0或是其他的，先更新一下
        self.refresh()

        pos_cur = self.rb_states[self.ee_idxs, :3]
        orn_cur = self.rb_states[self.ee_idxs, 3:7]

        # Solve for control (Operational Space Control)
        m_inv = torch.inverse(self.mm)
        m_eef = torch.inverse(self.j_eef @ m_inv @ torch.transpose(self.j_eef, 1, 2))
        orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
        orn_err = orientation_error(orn_des, orn_cur)

        pos_err = kp * (pos_des - pos_cur)
        dpose = torch.cat([pos_err, orn_err], -1)

        u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * self.mm @ self.dof_vel

        return  u
    
    def get_collision_forces(self):
        return self.contact_forces
    
    # ✅ 末端执行器碰撞力
    def get_ee_collision_info(self):
        self.refresh()
        ee_collision_forces = self.contact_forces[self.ee_idxs, :3]
        force_magnitudes = torch.norm(ee_collision_forces, dim=1)
        return {'force_magnitudes':force_magnitudes}
    
    def get_finger_center_position(self):
        left_finger_pos = self.rb_states[self.left_finger_idxs, :3]
        right_finger_pos = self.rb_states[self.right_finger_idxs, :3]
        center_pos = (left_finger_pos + right_finger_pos) / 2
        return center_pos
    
    # ✅ 末端执行器位置
    def get_ee_position(self):
        ee_pos = self.rb_states[self.ee_idxs, :3]
        return ee_pos

    # ✅ 末端执行器旋转（四元数）
    def get_ee_orientation(self):
        ee_orn = self.rb_states[self.ee_idxs, 3:7]  # 四元数 (x, y, z, w)
        return ee_orn

    # ✅ 末端执行器速度
    def get_ee_velocity(self):
        ee_vel = self.rb_states[self.ee_idxs, 7:10]
        return ee_vel

    # ✅ 末端执行器角速度
    def get_ee_angular_velocity(self):
        ee_ang_vel = self.rb_states[self.ee_idxs, 10:13]  # ωx, ωy, ωz
        return ee_ang_vel

    # ✅ 手指开合宽度（如果有夹爪）
    def get_fingers_width(self):
        # 举例: panda_finger_joint1 和 panda_finger_joint2
        left = self.dof_pos[:, 7, 0]
        right = self.dof_pos[:, 8, 0]
        width = left + right
        return width

    # ✅ 获取单个关节角

    # acotor类型
    def get_joint_angle(self, joint_index):
        return self.dof_pos[:, joint_index, 0]

    # ✅ 获取所有关节角
    def get_joint_angles(self):
        return self.dof_pos

    # ✅ 获取单个关节速度
    def get_joint_velocity(self, joint_index):
        return self.dof_vel[:, joint_index, 0]

    # ✅ 获取所有关节速度
    def get_joint_velocities(self):
        return self.dof_vel

    # ✅ 设置关节角度
    def set_joint_angles(self, target_joints):
        target = torch.tensor(target_joints, dtype=torch.float32, device=self.dof_pos.device)
        self.gym.set_dof_position_tensor(self.sim, gymtorch.unwrap_tensor(target))

    def set_joint_neutral(self,target_joint):
        if target_joint != None:
            self.set_joint_angles(target_joint)
        else:
            self.set_joint_angles(self.robot_mids)

    # ✅ 设置底座位姿
    def set_actor_pose(self, name, pos, orn,env_ids):
        transform = gymapi.Transform()
        # 把 Tensor 转为 list
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().tolist()
        if isinstance(orn, torch.Tensor):
            orn = orn.detach().cpu().tolist()
        transform.p = gymapi.Vec3(pos[0], pos[1], pos[2])
        transform.r = gymapi.Quat(orn[0], orn[1], orn[2], orn[3])
        for i in env_ids:
            actor_handle = self.gym.find_actor_handle(self.envs[i], name)
            self.gym.set_rigid_transform(self.envs[i], actor_handle, transform)

    def generate_random_ball_pose(self):
        ball_pose = gymapi.Transform()
        x = random.uniform(0.3, 0.7)
        y = random.uniform(-0.1, 0.1)
        z = 0.3
        ball_pose.p = gymapi.Vec3(x, y, z)
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        return ball_pose
    
    def get_ball_positions(self):
        ball_pose = self.rb_states[self.ball_idxs, :3]
        return ball_pose

    def create_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)



    def reset_joint_states(self, env_ids):
        """重置指定环境的关节状态到初始位置（GPU pipeline 友好：使用 Tensor API）
        
        Args:
            env_ids: 需要重置的环境ID，torch.Tensor类型
        """
        if env_ids is None or len(env_ids) == 0:
            return
        # 确保最新的 dof tensor 已获取
        self.gym.refresh_dof_state_tensor(self.sim)

        # Isaac Gym 的 DOF 状态张量按环境连续存储
        dofs_per_env = self.robot_num_dofs
        # 目标位姿/速度
        target_pos = torch.as_tensor(self.robot_mids, device=self.dof_states.device, dtype=self.dof_states.dtype)
        target_vel = torch.zeros_like(target_pos)

        for env_idx in env_ids.tolist():
            start = env_idx * dofs_per_env
            end = start + dofs_per_env
            # pos -> [:,0], vel -> [:,1]
            self.dof_states[start:end, 0] = target_pos
            self.dof_states[start:end, 1] = target_vel

        # 重新更新状态
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)


        # 回写整张 dof 状态张量（GPU pipeline 允许）
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        # 刷新张量视图
        self.refresh()
