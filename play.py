import os
import re
import argparse
from typing import Optional, List

# 先导入依赖 isaacgym 的配置与环境，再导入 torch 与 rsl_rl，避免导入顺序问题
from configs.Robot_config import FrankaReachCfg
from env.TaskRobotEnv import FrankaReachFixedPointGym

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO
from rsl_rl.utils import class_to_dict


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

    cfg = FrankaReachCfg()
    train_cfg = class_to_dict(rslCfgPPO())
    env =FrankaReachFixedPointGym(cfg)
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
        for _ in range(max_len):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play/evaluate trained policy for Franka Reach")
    parser.add_argument("--model", type=str, default=None, help="模型路径，默认自动选择 rsl_rl/logs 下最新 model_*.pt")
    parser.add_argument("--episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--stochastic", action="store_true", help="若提供则启用随机策略（若策略支持）")
    args = parser.parse_args()

    # 目前 get_inference_policy 返回确定性策略，--stochastic 预留
    eval_policy(model_path=args.model, episodes=args.episodes, deterministic=not args.stochastic)

