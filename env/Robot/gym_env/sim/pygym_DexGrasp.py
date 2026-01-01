import random
import time

import numpy as np

# gym应该要实现的接口
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import math
import sys

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
        self.sim_params.dt = 1.0 / 120.0
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
        # if getattr(args, 'use_gpu', False) or getattr(args, 'use_gpu_pipeline', False):
        #     compute_id = getattr(args, 'compute_device_id', 0)
        #     self.device = torch.device(f'cuda:{compute_id}') if torch.cuda.is_available() else torch.device('cpu')
        # else:
        #     self.device = torch.device('cpu')

        self.sim_device = args.sim_device            # 'cuda:0' / 'cuda:1' / 'cpu'  
        self.device = torch.device(self.sim_device)
        if self.sim_device.startswith("cuda"):
            self.compute_device_id = int(self.sim_device.split(":")[1])
        else:
            self.compute_device_id = -1

        # create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine,self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        
        self.enable_viewer = False
        self.viewer = None
        
        # create viewer
        if not getattr(self.args, 'headless', False):
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
                    
            
            
    def create_robot_asset(self,urdf_file,asset_root):
        # 创建模板
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        print("Loading asset '%s' from '%s'" % (urdf_file, asset_root))
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, urdf_file, asset_options)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        shape_props = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        for sp in shape_props:
            sp.friction = 1.8             # 动摩擦系数
            sp.rolling_friction = 0.0      # 滚动摩擦
            sp.torsion_friction = 0.0      # 扭转摩擦
            sp.restitution = 0.0           # 弹性（反弹）
        self.gym.set_asset_rigid_shape_properties(self.robot_asset, shape_props)

    def create_table_asset(self):
        # 创建模板
        table_dims = gymapi.Vec3(1, 1, 0.3)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    def create_box_asset(self):
        box_size = 0.05
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.thickness = 0.001
        self.box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        shape_props = self.gym.get_asset_rigid_shape_properties(self.box_asset)
        for sp in shape_props:
            sp.friction = 1.2             # 动摩擦系数
            sp.rolling_friction = 0.0      # 滚动摩擦
            sp.torsion_friction = 0.0      # 扭转摩擦
            sp.restitution = 0.0           # 弹性（反弹）
        self.gym.set_asset_rigid_shape_properties(self.box_asset, shape_props)

    def create_ball_asset(self):
        radius = 0.025
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        self.ball_asset = self.gym.create_sphere(self.sim, radius, asset_options)


    #后面接入参数，设置pd参数等等
    def set_dof_states_and_propeties(self, control_type):

        # set default DOF states
        self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"][:7] = self.robot_mids[:7]
        self.default_dof_state["pos"][7:15] = 0
        self.default_dof_state["pos"][15:16] = 1
        self.default_dof_state["pos"][16:] = 0
        self.default_dof_pos = torch.tensor(self.default_dof_state["pos"],dtype=torch.float32,device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # set DOF control properties (except grippers)
        # self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        # self.robot_dof_props["stiffness"][:7].fill(0.0)
        # self.robot_dof_props["damping"][:7].fill(0.0)
    
        # # set DOF control properties for grippers
        # self.robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        # self.robot_dof_props["stiffness"][7:].fill(0)
        # self.robot_dof_props["damping"][7:].fill(0)
        self.torque_limits = torch.tensor(
            self.robot_dof_props["effort"],
            device=self.device,
            dtype=torch.float32
        )
        self.torque_limits = self.torque_limits.unsqueeze(0) 

        
        if control_type == "effort" :
            self.robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
        elif control_type == "position" :
            self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            self.robot_dof_props["stiffness"][:7].fill(400) #参数需要修改
            self.robot_dof_props["damping"][:7].fill(40)

            self.robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
            self.robot_dof_props["stiffness"][7:].fill(50)
            self.robot_dof_props["damping"][7:].fill(5)




    def create_envs_and_actors(self,num_envs,base_pos,base_orn):
        # 首先是根据 base_pos和base_orn创建对应的 gyapi.Transform()
        pose=gymapi.Transform()
        pose.p =gymapi.Vec3(base_pos[0],base_pos[1],base_pos[2])
        pose.r=gymapi.Quat(base_orn[0],base_orn[1],base_orn[2],base_orn[3])

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.7, 0.0, 0.15)


        self.num_envs=num_envs
        self.envs=[]

        self.table_handles=[]
        self.table_idxs=[]
        self.box_handles=[]
        self.box_idxs=[]
        self.root_box_idxs=[]

        self.ee_handles=[]
        self.ee_idxs=[]

        self.hand_base_idxs=[]

        self.finger1_idxs=[]
        self.finger2_idxs=[]
        self.finger3_idxs=[]
        self.finger4_idxs=[]  
        self.finger5_idxs=[]
        self.body_link3_idxs=[]
        self.body_link4_idxs=[]
        self.body_link5_idxs=[]
        self.body_link6_idxs=[]

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

            box_goal_pose = self.generate_random_box_goal_pose()

            table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 1)
            self.table_handles.append(table_handle)
            table_idx = self.gym.find_actor_rigid_body_index(env, table_handle, "table", gymapi.DOMAIN_SIM)
            self.table_idxs.append(table_idx)
            

            box_handle = self.gym.create_actor(env, self.box_asset, box_goal_pose, "box", i, 0)
            red_color = gymapi.Vec3(1.0, 0.0, 0.0)  
            self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, red_color)
            self.box_handles.append(box_handle)
            box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            self.box_idxs.append(box_idx)
            root_box_idx = self.gym.get_actor_index(env,box_handle,gymapi.DOMAIN_SIM)
            self.root_box_idxs.append(root_box_idx)

            # Add franka
            robot_handle = self.gym.create_actor(env, self.robot_asset, pose, "franka", i, 1)

            # Set initial DOF states
            self.gym.set_actor_dof_states(env, robot_handle, self.default_dof_state, gymapi.STATE_ALL)

            # Set DOF control properties
            self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

            # Get inital ee pose
            ee_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle, "hand_base_link")
            self.ee_handles.append(ee_handle)
            ee_pose = self.gym.get_rigid_transform(env, ee_handle)
            self.init_pos_list.append([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z])
            self.init_orn_list.append([ee_pose.r.x, ee_pose.r.y, ee_pose.r.z, ee_pose.r.w])

            hand_base_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base_link", gymapi.DOMAIN_SIM)
            self.hand_base_idxs.append(hand_base_idx)
            
            #get finger pose
            finger1_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "thumb_distal", gymapi.DOMAIN_SIM)
            self.finger1_idxs.append(finger1_idx)
            finger2_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "index_distal", gymapi.DOMAIN_SIM)
            self.finger2_idxs.append(finger2_idx)
            finger3_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "middle_distal", gymapi.DOMAIN_SIM)
            self.finger3_idxs.append(finger3_idx)
            finger4_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "ring_distal", gymapi.DOMAIN_SIM)
            self.finger4_idxs.append(finger4_idx)
            finger5_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "pinky_distal", gymapi.DOMAIN_SIM)
            self.finger5_idxs.append(finger5_idx)

            body_link3_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link3", gymapi.DOMAIN_SIM)
            self.body_link3_idxs.append(body_link3_idx)
            body_link4_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link4", gymapi.DOMAIN_SIM)
            self.body_link4_idxs.append(body_link4_idx)
            body_link5_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link5", gymapi.DOMAIN_SIM)
            self.body_link5_idxs.append(body_link5_idx)
            body_link6_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link6", gymapi.DOMAIN_SIM)
            self.body_link6_idxs.append(body_link6_idx)

            

            # Get global index of ee in rigid body state tensor
            ee_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base_link", gymapi.DOMAIN_SIM)
            self.ee_idxs.append(ee_idx)


    def set_camera(self):
        # Point camera at middle env
        if getattr(self.args, 'headless', False):
            return
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def pre_simulate(self,num_envs,asset_root,asset_file,base_pos,base_orn,control_type):
        self.create_plane()
        self.create_robot_asset(asset_file,asset_root)
        self.create_table_asset()
        self.create_box_asset()
        self.create_ball_asset()

        # get joint limits and ranges for Franka
        self.robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        self.robot_lower_limits = self.robot_dof_props['lower']
        self.robot_upper_limits = self.robot_dof_props['upper']
        robot_ranges = self.robot_upper_limits - self.robot_lower_limits
        # 设置一下robot_mids,可能是用来初始化的作用，这个地方稍微记忆一下.
        self.robot_mids = 0.5 * (self.robot_upper_limits + self.robot_lower_limits)
        self.robot_num_dofs = len(self.robot_dof_props)

        self.set_dof_states_and_propeties(control_type)
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

        self._root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self._root_states)  

        # 拆分位置与速度分量
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)

        # Jacobian entries for end effector
        self.ee_index = self.gym.get_asset_rigid_body_dict(self.robot_asset)["hand_base_link"]
        self.j_eef = self.jacobian[:, self.ee_index - 1, :]

        # Prepare mass matrix tensor
        self._massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(self._massmatrix)
        self.refresh()
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()


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
        self.refresh()
    # Step rendering (skip when headless)
        self.render()

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
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def ee_pos_to_torque(self,pos_des,orn_des):
        # 由末端位置控制,由雅可比矩阵等计算出对应的力矩
        kp = 5
        kv = 2 * math.sqrt(kp)
        # 使用 之前，先要同步一下，刚创建的时候可能数据为0或是其他的，先更新一下
        self.refresh()

        pos_cur = self.rb_states[self.ee_idxs, :3]
        orn_cur = self.rb_states[self.ee_idxs, 3:7]

        mm_arm = self.mm[:, :7, :7]
        j_eef_arm = self.j_eef[:, :, :7]
        dof_vel_arm = self.dof_vel[:, :7, 0].unsqueeze(-1)

        # Solve for control (Operational Space Control)
        m_inv = torch.inverse(mm_arm)
        m_eef = torch.inverse(j_eef_arm @ m_inv @ torch.transpose(j_eef_arm, 1, 2))
        orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
        orn_err = orientation_error(orn_des, orn_cur)

        pos_err = kp * (pos_des - pos_cur)
        dpose = torch.cat([pos_err, orn_err], -1)

        u = torch.transpose(j_eef_arm, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm_arm @ dof_vel_arm
        return  u
    
    def body_joint_to_torque(self, body_displacement, body_joint_pos, body_joint_vel, kp , kv):
        #u = kp * (body_displacement + self.default_dof_pos[:,:7] - body_joint_pos) - kv * body_joint_vel
        u = kp * body_displacement  - kv * body_joint_vel
        u = torch.clamp(u, -self.torque_limits[:,:7], self.torque_limits[:,:7])
        return u

    def hand_joint_to_torque(self, hand_displacement, hand_joint_pos, hand_joint_vel, kp, kv):
        #u = kp * (hand_displacement + self.default_dof_pos[:,7:] - hand_joint_pos) - kv * hand_joint_vel
        u = kp * hand_displacement  - kv * hand_joint_vel
        u = torch.clamp(u, -self.torque_limits[:,7:], self.torque_limits[:,7:])
        return u
    
    def body_joint_to_pos(self, body_displacement, body_joint_pos):
        u = body_joint_pos + body_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[:7],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[:7],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u
    
    def hand_joint_to_pos(self, hand_displacement, hand_joint_pos):
        u = hand_joint_pos + hand_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[7:],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[7:],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u

    
    def get_collision_forces(self):
        return self.contact_forces
    
    # ✅ 末端执行器碰撞力
    def get_ee_collision_info(self):
        self.refresh()
        ee_collision_forces = self.contact_forces[self.ee_idxs, :3]
        force_magnitudes = torch.norm(ee_collision_forces, dim=1)
        return {'force_magnitudes':force_magnitudes}
    
    def get_hand_base_pos(self):
        # finger1_pos = self.rb_states[self.finger1_idxs, :3]
        # finger2_pos = self.rb_states[self.finger2_idxs, :3]
        # finger3_pos = self.rb_states[self.finger3_idxs, :3]
        # finger4_pos = self.rb_states[self.finger4_idxs, :3]
        # finger5_pos = self.rb_states[self.finger5_idxs, :3]
        # center_pos = (finger1_pos + finger2_pos + finger3_pos + finger4_pos + finger5_pos) / 5.0
        hand_base_pos = self.rb_states[self.hand_base_idxs, :3]
        return hand_base_pos
    
    def get_two_fingers_mid_point(self):
        finger1_pos = self.rb_states[self.finger1_idxs, :3]
        finger2_pos = self.rb_states[self.finger2_idxs, :3]
        center_pos = (finger1_pos + finger2_pos) / 2.0
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

    def get_hand_to_object_distance(self):
        hand_base_pos = self.get_hand_base_pos()
        box_goal_pos = self.get_obj_position()
        distance = hand_base_pos - box_goal_pos
        return distance

    # ✅ 获取单个关节角

    # acotor类型
    def get_hand_joint_pos(self):
        hand_joints_pos = self.dof_pos[:, 7:18, 0] 
        return hand_joints_pos
    
    def get_hand_joint_vel(self):
        hand_joints_vel = self.dof_vel[:, 7:18, 0] 
        return hand_joints_vel
    
    def get_body_joint_pos(self):
        body_joints_pos = self.dof_pos[:, :7, 0]
        return body_joints_pos
    
    def get_body_joint_vel(self):
        body_joints_vel = self.dof_vel[:, :7, 0]
        return body_joints_vel

    # ✅ 获取所有关节角
    def get_joint_pos(self):
        joint_pos = self.dof_pos[:, :, 0]
        return joint_pos

    # ✅ 获取单个关节速度
    def get_joint_velocity(self, joint_index):
        return self.dof_vel[:, joint_index, 0]

    # ✅ 获取所有关节速度
    def get_joint_vel(self):
        joint_vel = self.dof_vel[:, :, 0]
        return joint_vel

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

    def generate_random_box_goal_pose(self):
        box_goal_pose = gymapi.Transform()
        x = random.uniform(0.5, 0.7)
        y = random.uniform(-0.1, 0.1)
        z = 0.325
        box_goal_pose.p = gymapi.Vec3(x, y, z)
        box_goal_pose.r = gymapi.Quat(0, 0, 0, 1)
        return box_goal_pose
    
    def get_obj_position(self):
        box_goal_pose = self.root_states[self.root_box_idxs, :3]
        return box_goal_pose
    
    def get_obj_quaternion(self):
        box_goal_quat = self.root_states[self.root_box_idxs, 3:7]
        return box_goal_quat
    
    def get_finger_collision_info(self):
        finger1_pos_z = self.rb_states[self.finger1_idxs, 2]
        finger2_pos_z = self.rb_states[self.finger2_idxs, 2]
        finger3_pos_z = self.rb_states[self.finger3_idxs, 2]
        finger4_pos_z = self.rb_states[self.finger4_idxs, 2]
        finger5_pos_z = self.rb_states[self.finger5_idxs, 2]
        table_pos_z = 0.3  
        collision_finger1 = finger1_pos_z < table_pos_z
        collision_finger2 = finger2_pos_z < table_pos_z     
        collision_finger3 = finger3_pos_z < table_pos_z
        collision_finger4 = finger4_pos_z < table_pos_z
        collision_finger5 = finger5_pos_z < table_pos_z
        collision = collision_finger1 | collision_finger2 | collision_finger3 | collision_finger4 | collision_finger5
        return {
            'collision_flags': collision
        }
    
    def get_body_collision_info(self):#修改为臂pos低于手pos
        # body_link3_pos_z = self.rb_states[self.body_link3_idxs, 2]
        # body_link4_pos_z = self.rb_states[self.body_link4_idxs, 2]
        body_link5_pos_z = self.rb_states[self.body_link5_idxs, 2]
        body_link6_pos_z = self.rb_states[self.body_link6_idxs, 2]
        fingger_pos = self.get_hand_base_pos()
        fingger_pos_z = fingger_pos[:, 2]
        # collision_body3 = body_link3_pos_z < fingger_pos_z
        # collision_body4 = body_link4_pos_z < fingger_pos_z    
        collision_body5 = body_link5_pos_z < fingger_pos_z
        collision_body6 = body_link6_pos_z < fingger_pos_z
        collision = collision_body5 | collision_body6
        return {
            'collision_flags': collision
        }
    
    #物体重置条件
    def get_object_reset_info(self):
        box_pos_z = self.rb_states[self.box_idxs, 2]
        table_pos_z = 0.3
        reset_obj = box_pos_z < table_pos_z
        return {
            'reset_obj': reset_obj
        }

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

        for env_idx in env_ids.tolist():
            start = env_idx * dofs_per_env
            end = start + dofs_per_env
            # pos -> [:,0], vel -> [:,1]
            self.dof_states[start:end, 0] = self.initial_dof_states[start:end, 0]
            self.dof_states[start:end, 1] = self.initial_dof_states[start:end, 1]

        # 重新更新状态
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)


        # 回写整张 dof 状态张量（GPU pipeline 允许）
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        # 刷新张量视图
        self.refresh()
        
        
    def reset_object_states(self, env_ids):
        # if env_ids is None or len(env_ids) == 0:
        #     return
        # # 确保最新的 dof tensor 已获取

        # for env_idx in env_ids.tolist():

        #     self.root_states[self.box_idxs, :3][env_idx] = self.initial_root_states[self.box_idxs, :3][env_idx]
        #     self.root_states[self.box_idxs, 3:7][env_idx] = self.initial_root_states[self.box_idxs, 3:7][env_idx]
        #     self.root_states[self.box_idxs, 7:13][env_idx] = torch.zeros(6, device=self.root_states.device)

        # self.gym.set_actor_root_state_tensor(self.sim)
        # self.refresh()
        if env_ids is None or len(env_ids) == 0:
            return

        for env_idx in env_ids.tolist():
            reset_obj_idxs = self.root_box_idxs[env_idx]   # ✅ 关键一步

            self.root_states[reset_obj_idxs, 0:3] = self.initial_root_states[reset_obj_idxs, 0:3]
            self.root_states[reset_obj_idxs, 3:7] = self.initial_root_states[reset_obj_idxs, 3:7]
            self.root_states[reset_obj_idxs, 7:13] = torch.zeros(6, device=self.root_states.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.refresh()

    def get_num_dofs(self):
        robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        robot_num_dofs = len(robot_dof_props)
        return robot_num_dofs
    
    def get_dof_names(self):
        dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        for i, name in enumerate(self.dof_names):
            print(i, name)
        return dof_names
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer" and evt.value > 0:
                    self.enable_viewer = not self.enable_viewer

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def check_reset_events(self):
        reset_events = {}

        finger_info = self.get_finger_collision_info()
        reset_events['finger_collision'] = finger_info['collision_flags']

        body_info = self.get_body_collision_info()
        reset_events['body_collision'] = body_info['collision_flags']

        obj_info = self.get_object_reset_info()
        reset_events['obj_reset'] = obj_info['reset_obj']

        return reset_events

    def get_rigid_body_x_axis_world(self):
        """
        获取任意刚体的 x轴在世界坐标系中的方向

        Args:
            body_indices: list[int] 或 torch.Tensor，刚体在 rb_states 中的 index

        Returns:
            -x_axis_world: (N, 3)
        """

        self.refresh()

        quat = self.rb_states[self.hand_base_idxs, 3:7]
        quat = quat / torch.norm(quat, dim=1, keepdim=True)

        # 掌心法向量x轴
        x_local = torch.tensor(
            [1.0, 0.0, 0.0],
            device=quat.device
        ).expand(quat.shape[0], 3)

        x_world = quat_rotate(quat, x_local)

        return x_world
    

    def get_finger_z_distance(self):
        finger1_pos_z = self.rb_states[self.finger1_idxs, 2]
        finger2_pos_z = self.rb_states[self.finger2_idxs, 2]
        distance = abs(finger1_pos_z - finger2_pos_z)

        return distance

