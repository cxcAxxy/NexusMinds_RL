import os
import numpy as np
from datetime import datetime
import time
import sys
import torch
import mujoco
import mujoco.viewer
from typing import Optional, List
import argparse
import re
import math
import torch
from spatialmath import SE3
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R

from env.core import RobotTaskEnv
from env.Robot.gym_env.instance.franka import Franka
from env.Task import *
# 首先导入包含 Isaac Gym 的配置和环境模块
# from configs.Robot_config import FrankaReachMujocoCfg
# from env.TaskRobotEnv import FrankaReachRandPointsMujoco
from env.core import RobotTaskEnv
from env.Robot.MuJoCo_env.sim.pymujoco import Mujoco

# 然后导入可能包含 PyTorch 的 rsl_rl 模块
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO, envCfg
from rsl_rl.utils import class_to_dict
import warnings

# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

#/home/sxy/下载/scence.xml
#/home/sxy/桌面/mujoco-3.3.7-linux-x86_64/mujoco-3.3.7/bin/scene.xml

class FrankaReachRandPointsMujoco(RobotTaskEnv):
    def __init__(self, cfg) -> None:
        sim = Mujoco(cfg.mujococfg)
        robot = Franka(sim, cfg.robotcfg)
        task = Reach_random_points(sim, cfg.taskcfg)

        # 调用父类初始化，传入机器人、任务和配置
        super().__init__(
            robot,
            task,
            cfg)


class MujocoCfg:
    """仿真器配置"""
    def __init__(self):
        # 默认值
        self.neutral_joint_values = [0,0,0,0,0,0]
        self.render_mode = "human"
        self.n_substeps =1
        self.model_path = "/home/sxy/桌面/NexusMInds_RL-main/env/assets/xml/franka_emika_panda/scene.xml"




class MujocoRobotCfg:
    """机械臂配置"""
    def __init__(self):
        # 控制相关参数
        self.control_type = "ee"      
        self.block_gripper = True 
        self.num_actions = 3            
        self.num_obs = 9
        self.num_envs = 1 # 修改为与其他配置一致
        self.control_type_sim = "effort"             

        # 模型路径与姿态
        self.asset = "/home/cxc/Desktop/NexusMinds_RL/env/assets"
        self.robot_files = "xml/franka_emika_panda/scene.xml"
        # 每个机器人的初始位置是一样的吗
        self.base_pose = [0,0,0]  # 每个环境的机器人位置
        self.base_orn = [0,0,0,1] # 每个环境的机器人姿态

        self.ee_link = "hand"
        self.headless = "False"
        self.control_decimation = 6
        self.action_low = -1
        self.action_high = 1


class TaskCfg:
    """Franka Reach 任务配置"""
    def __init__(self):
        self.name = "Reach"
        self.device = torch.device("cpu")
        self.num_envs = 1

        self.reward_type = "dense"
        self.distance_threshold = 0.05

        self.goal_range = 1
        self.get_ee_position = None


class AllCfg:
    """环境总体配置"""
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.num_achieved_goal = 3
        self.num_desired_goal = 3
        self.max_episode_length = 200
        self.max_episode_length_s = 4.0  # 秒数形式（用于日志统计）
        self.decimation = 4  
        self.control_type_sim = "effort"  



class FrankaReachMujocoCfg:
    """mujoco总配置类"""
    def __init__(self):
        self.mujococfg = MujocoCfg()
        self.robotcfg = MujocoRobotCfg()
        self.taskcfg = TaskCfg()
        self.all = AllCfg()


def _find_latest_checkpoint(log_dir: str) -> Optional[str]:
    if not os.path.isdir(log_dir):
        return None
    pattern = re.compile(r"model_(\d+)\.pt$")
    cands: List[str] = []
    for fn in os.listdir(log_dir):
        if pattern.search(fn):
            cands.append(fn)
    if not cands:
        return None
    # 取最大编号
    cands.sort(key=lambda x: int(pattern.search(x).group(1)))
    return os.path.join(log_dir, cands[-1])


def eval_policy(model_path: Optional[str] = None, episodes: int = 10, deterministic: bool = True):

    cfg = FrankaReachMujocoCfg()
    train_cfg = class_to_dict(rslCfgPPO())
    env =FrankaReachRandPointsMujoco(cfg)
    runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=None, device=str(env.device))

    if model_path is None:
        model_path = _find_latest_checkpoint(log_dir="logs")
        if model_path is None:
            raise FileNotFoundError("未找到 checkpoint，请先训练或通过 --model 指定路径")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"无效的模型路径: {model_path}")

    # 仅加载权重做推理
    runner.load(model_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    num_envs = env.num_envs
    max_len = int(env.max_episode_length)

    successes: List[bool] = []
    min_dists: List[float] = []

    # 3) 分批评估（每批 num_envs 个并行环境）
    total_done = 0
    while total_done < episodes:
        # 重置一批环境
        obs, _ = env.reset()
        # 记录每个并行环境在本回合是否成功、最小距离
        batch_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        batch_min_dist = torch.full((num_envs,), float("inf"), device=env.device)

        # 固定长度回合，不依赖 dones（环境内部会在超时后重置）
        for _ in range(1000):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)

            # 成功判定与距离统计
            achieved = env.get_achieved_goal_obs()
            desired = env.get_desired_goal_obs()
            print("------检验-----")
            print("Achieved Goals:\n", achieved)
            print("Desired Goals:\n", desired)
            print("-----------------")
            dist = torch.norm(achieved - desired, dim=-1)
            batch_min_dist = torch.minimum(batch_min_dist, dist)

            step_success = env.task.is_success(achieved, desired)
            batch_success |= step_success

        # 汇总本批结果（可能超过所需的 episodes，截断即可）
        remain = episodes - total_done
        take = min(remain, num_envs)
        successes.extend(batch_success[:take].bool().tolist())
        min_dists.extend(batch_min_dist[:take].float().tolist())
        total_done += take

    success_rate = sum(1 for s in successes if s) / len(successes)
    avg_min_dist = sum(min_dists) / len(min_dists) if min_dists else float("nan")

    print("===== Evaluation =====")
    print(f"Checkpoint: {model_path}")
    print(f"Episodes:   {episodes}")
    print(f"Success@{env.task.distance_threshold:.3f}m: {success_rate*100:.2f}%")
    print(f"Avg min distance: {avg_min_dist:.4f} m (越小越好)")
# 简单测试函数
def best_ik_pose(robot_chain, target_position, target_quat, trials=20):
    """
    输入末端目标位置和姿态，返回最优关节角。

    Args:
        env: 仿真环境对象，用于获取实际末端位姿
        robot_chain: 包含 forward_kinematics 和 inverse_kinematics 方法的机械臂链
        target_position: np.array([x,y,z])
        target_quat: np.array([w,x,y,z])
        trials: 随机初值尝试次数

    Returns:
        best_q: 最优关节角
        best_err: 最小位置+姿态误差
    """
    best_q = None
    best_err = float('inf')

    # 可动关节索引和上下限
    active_idx = [i for i, a in enumerate(robot_chain.active_links_mask) if a]
    lower_bounds = [robot_chain.links[i].bounds[0] for i in active_idx]
    upper_bounds = [robot_chain.links[i].bounds[1] for i in active_idx]

    for _ in range(trials):
        # 随机初值
        q0_full = np.zeros(len(robot_chain.links))
        for idx, i in enumerate(active_idx):
            q0_full[i] = np.random.uniform(lower_bounds[idx], upper_bounds[idx])

        # 逆运动学求解
        q_ik = robot_chain.inverse_kinematics(
            target_position,
            initial_position=q0_full
        )

        # 正向运动学得到末端位姿
        fk_matrix = robot_chain.forward_kinematics(q_ik)  # 4x4齐次矩阵
        fk_pos = fk_matrix[:3, 3]

        # 姿态误差
        fk_rot = R.from_matrix(fk_matrix[:3, :3])
        target_rot = R.from_quat(target_quat)
        # 相对旋转四元数
        q_rel = fk_rot.inv() * target_rot
        angle_err = np.linalg.norm(q_rel.as_rotvec())  # 旋转向量角度误差

        # 位置误差
        pos_err = np.linalg.norm(fk_pos - target_position)

        # 总误差：位置 + 姿态（可调比例）
        total_err = pos_err + angle_err

        if total_err < best_err:
            best_err = total_err
            best_q = q_ik

    return best_q, best_err

def test_basic_functionality():

    urdf = "/home/sxy/桌面/NexusMInds_RL-main/env/assets/urdf/franka_description/robots/franka_panda.urdf"


    cfg = FrankaReachMujocoCfg()
    train_cfg = class_to_dict(rslCfgPPO())

    
    # 使用正确的环境类和配置
    env = FrankaReachRandPointsMujoco(cfg)
    # goal_rot = torch.tensor([3.8268e-01, -2.7534e-08, -1.1405e-08,  9.2388e-01], device="cpu")
    # env.sim.evaluate_ik_accuracy(n_samples=30)
    
    target_position = [0.2033, 0,  0.5158]
    target_quat = [1, 0, 0, 0]

    hand_body_id = env.sim.model.body('hand').id
   
    qpos_solution  = env.sim.solve_ik_6d(
        pos_target=np.array(target_position),
        quat_target=np.array(target_quat),
        n_iter=500,
        eps_pos=1e-4,
        eps_rot=1e-3,
        weight_rot=0.2,
        max_dq=0.1
    )
    qpos_solution = np.array(qpos_solution).flatten()
    print("IK 关节角解：", qpos_solution)
    # env.sim.data.qpos[:] = qpos_solution
    # mujoco.mj_forward(env.sim.model, env.sim.data)

    # fk_pos = np.array(env.sim.data.body(hand_body_id).xpos)
    # fk_quat = np.array(env.sim.data.body(hand_body_id).xquat)
    # print("FK 验证位置：", fk_pos)
    # print("FK 验证四元数：", fk_quat)


    
    env.sim.set_joint_angles(qpos_solution[:7])

    for _ in range(1000):
        env.sim.mjstep()
        env.sim.viewer.sync()
        time.sleep(0.01)

    print("final pos:", env.sim.get_ee_position())
    print("final orn:", env.sim.get_ee_orientation())

    
    while env.sim.viewer.is_running():
        env.sim.mjstep()
        env.sim.viewer.sync()
        time.sleep(0.01)


    
    

if __name__ == "__main__":
    test_basic_functionality()
    # parser = argparse.ArgumentParser(description="Play/evaluate trained policy for Franka Reach")
    # parser.add_argument("--model", type=str, default=None, help="模型路径，默认自动选择 rsl_rl/logs 下最新 model_*.pt")
    # parser.add_argument("--episodes", type=int, default=10, help="评估回合数")
    # parser.add_argument("--stochastic", action="store_true", help="若提供则启用随机策略（若策略支持）")
    # args = parser.parse_args()

    # # 目前 get_inference_policy 返回确定性策略，--stochastic 预留
    # eval_policy(model_path=args.model, episodes=args.episodes, deterministic=not args.stochastic)
