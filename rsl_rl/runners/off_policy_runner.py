# import os
# import torch
# import numpy as np
# from rsl_rl.modules import TD3ActorCritic
# from rsl_rl.algorithms import DDPG

# # ================= OffPolicyRunner =================
# class OffPolicyRunner:

#     def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
#         self.env = env
#         self.cfg = train_cfg["runner"]
#         self.alg_cfg = train_cfg["algorithm"]
#         self.policy_cfg = train_cfg["policy"]
#         self.device = device
#         self.log_dir = log_dir


#         obs_dim = int(env.observation_space.shape[0])
#         act_dim = int(env.action_space.shape[0])
#         self.num_envs = int(getattr(env, 'num_envs', 1))

#         # 连续动作缩放
#         self._action_high = np.asarray(env.action_space.high, dtype=np.float32)
#         self._action_low = np.asarray(env.action_space.low, dtype=np.float32)
#         self._max_action = np.asarray(env.action_space.high, dtype=np.float32)

#         # ActorCritic 模型
#         self.actor_critic = TD3ActorCritic(
#             num_actor_obs=obs_dim,
#             num_critic_obs=obs_dim,
#             num_actions=act_dim,
#             **self.policy_cfg
#         ).to(self.device)

#         # 算法
#         self.alg = DDPG(self.actor_critic, device=self.device, **self.alg_cfg)

#         self.alg.init_storage(
#                 buffer_size=int(self.cfg["buffer_size"]),
#                 obs_shape=[obs_dim],
#                 act_shape=[act_dim]
#             )
        
#         self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
#         self.save_interval = int(self.cfg["save_interval"])
#         self.start_random_steps = int(self.cfg("start_random_steps"))
#         self.tot_timesteps = 0
#         self.current_learning_iteration = 0

#     # ================== 主训练循环 ==================
#     def learn(self, num_learning_iterations, eval_env=None, writer=None):
#         obs, info  = self.env.reset()
#         obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

#         global_step = 0
#         for iter_idx in range(num_learning_iterations):
#             # ---------------- 收集数据 ----------------
#             for _ in range(self.num_steps_per_env):
#                 if global_step < self.start_random_steps:
#                     # 纯随机探索填充 buffer（环境动作尺度）
#                     action_env = self.env.action_space.sample()
#                     # 转为策略尺度 [-1, 1] 用于存储
#                     action_policy = np.clip(action_env / self._max_action, -1.0, 1.0)
#                 else:
#                     with torch.no_grad():
#                         action_policy = self.actor_critic.act_with_noise(obs_t).cpu().numpy()

#                     action_env = np.clip(action_policy * self._max_action,
#                                         self._action_low, self._action_high)

#                 next_obs, reward, terminated, truncated, info = self.env.step(action_env)
#                 done_env = bool(terminated or truncated)
#                 done_for_buffer = bool(terminated)  # 时间截断不作为终止，允许 bootstrap

#                 # 存储 transition：存策略尺度动作，保证与更新时 actor 输出一致
#                 self.alg.store_transition(
#                     obs_t.detach().cpu().numpy(),
#                     action_policy,
#                     float(reward),
#                     done_for_buffer,
#                     next_obs
#                 )
#                 obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
#                 global_step += 1

#                 if done_env:
#                     obs, info = self.env.reset()
#                     obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

#             # ---------------- 更新算法 ----------------
#             if global_step >= self.start_random_steps:
#                 self.alg.update()
#             self.current_learning_iteration += 1
#             self.tot_timesteps += self.num_steps_per_env * self.num_envs

#             # ---------------- 评估 ----------------
#             if eval_env is not None and (iter_idx % 10 == 0):
#                 score = self._evaluate(eval_env)
#                 print(f"Iter: {iter_idx}, Eval Reward: {score:.2f}")
#                 if writer is not None:
#                     writer.add_scalar("eval_reward", score, iter_idx)

#             # ---------------- 保存模型 ----------------
#             if self.log_dir is not None and (iter_idx % self.save_interval == 0):
#                 self.save_model(iter_idx)

#         # ---------------- 结束清理 ----------------
#         self.env.close()
#         if eval_env is not None:
#             eval_env.close()
#         if writer is not None:
#             writer.close()

#     # ================== 评估 ==================
#     def _evaluate(self, env, turns=3):
#         total_reward = 0.0
#         action_high = np.asarray(env.action_space.high, dtype=np.float32)
#         action_low = np.asarray(env.action_space.low, dtype=np.float32)
#         for _ in range(turns):
#             obs, info = env.reset()
#             done = False
#             while not done:
#                 with torch.no_grad():
#                     obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
#                     action = self.actor_critic.act(obs_t)
#                 action_np = action.detach().cpu().numpy()
#                 action_env = action_np * action_high
#                 action_env = np.clip(action_env, action_low, action_high)
#                 action_env = action_env.astype(env.action_space.dtype, copy=False)

#                 obs, reward, terminated, truncated, info = env.step(action_env)
#                 done = bool(terminated or truncated)

#                 total_reward += float(reward)
#         return total_reward / turns

#     # ================== 保存模型 ==================
#     def save_model(self, iter_idx):
#         if self.log_dir is None:
#             return
#         os.makedirs(self.log_dir, exist_ok=True)
#         save_path = os.path.join(self.log_dir, f"model_{iter_idx}.pt")
#         torch.save(self.actor_critic.state_dict(), save_path)
#         print(f"[Model Saved] -> {save_path}")


import os
import torch
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.modules import DDPGActorCritic
from rsl_rl.algorithms import DDPG
from rsl_rl.env import VecEnv
import statistics
import time


class OffPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.log_dir = log_dir
        self.env = env

        num_critic_obs = self.env.num_obs
        # ActorCritic 模型
        self.actor_critic = DDPGActorCritic(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)

        # 算法
        self.alg = DDPG(self.actor_critic, device=self.device, **self.alg_cfg)

        self.alg.init_storage(
            buffer_size=int(self.cfg["max_size"]),
            num_envs=self.env.num_envs,
            obs_shape=[self.env.num_obs],
            act_shape=[self.env.num_actions]
        )

        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])# 每个环境每次采集多少步数据
        self.save_interval = int(self.cfg["save_interval"])# 多久保存一次模型
        self.start_random_steps = int(self.cfg["start_random_steps"])# 在开始训练前采集多少步的随机数据
        self.tot_timesteps = 0
        self.current_learning_iteration = 0# 当前学习迭代次数

        _, _ = self.env.reset()

        # TensorBoard
        self.writer = None
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    # ================== 主训练循环 ==================
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length)) #让每个环境的 episode 从随机进度开始

        obs_t = self.env.get_observations()
        obs_t = obs_t.to(self.device) #训练数据统一搬到一个设备上
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        global_step = 0 

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations #断点续训
        for it in range(self.current_learning_iteration, tot_iter): 
            start = time.time()
            # ---------------- 收集数据 ----------------
            with torch.inference_mode():#用于在 推理/评估阶段 临时关闭梯度计算和某些训练相关的状态，从而提高性能和节省显存。
                for _ in range(self.num_steps_per_env):
                    if global_step < self.start_random_steps:
                        action = self.actor_critic.sample_random_action(self.env.num_envs)  # 随机采样
                    else:
                        with torch.no_grad():
                            action = self.actor_critic.act_with_noise(obs_t) # 带噪声采样

                    next_obs, privileged_obs, reward, done_env, infos = self.env.step(action)
                    next_obs, critic_obs, rewards, dones = next_obs.to(self.device), privileged_obs.to(self.device), reward.to(self.device), done_env.to(self.device)

                    # 储存这里需要再封装的好一些，类比rsl_rl！！！ 
                    # if 'time_outs' in infos:
                    #     self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
                    # 存储 transition
                    self.alg.store_transition(
                        obs_t,
                        action,
                        rewards,
                        dones,
                        next_obs
                    )

                    obs_t = next_obs
                    global_step += 1

                    # 记录训练奖励和长度
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop


            # ---------------- 更新算法 ----------------
            if global_step >= self.start_random_steps:
                critic_loss, actor_loss, noise_std = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            ep_infos.clear()

            # ---------------- 保存模型 ----------------
            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir is not None and (it % self.save_interval == 0):
                self.save_model(it)
        self.current_learning_iteration += num_learning_iterations


    # ================== 保存模型 ==================
    def save_model(self, iter_idx):
        if self.log_dir is None:
            return
        save_path = os.path.join(self.log_dir, f"model_{iter_idx}.pt")
        torch.save(self.actor_critic.state_dict(), save_path)
        print(f"[Model Saved] -> {save_path}")

    # ================== 记录日志 ==================
    def log(self, locs, critic_loss=None, actor_loss=None, noise_std=None, width=80, pad=35):
        """
        打印并记录训练日志
        locs: locals()，包含当前迭代所有局部变量
        critic_loss, actor_loss, noise_std: 来自 self.alg.update() 的训练指标
        """
        # 累计步数
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

        # 时间统计
        iteration_time = locs['collection_time'] + locs['learn_time']
        self.tot_time += iteration_time

        # FPS 计算
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)

        # Episode 信息
        ep_string = ""
        if 'ep_infos' in locs and locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                tensor_list = []
                for ep_info in locs['ep_infos']:
                    val = ep_info[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor([val], device=self.device)
                    elif val.dim() == 0:
                        val = val.unsqueeze(0)
                    tensor_list.append(val.to(self.device))
                mean_val = torch.cat(tensor_list).mean().item()
                if self.writer:
                    self.writer.add_scalar(f'Episode/{key}', mean_val, locs['it'])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {mean_val:.4f}\n"

        # DDPG 相关指标
        value_loss = critic_loss if critic_loss is not None else 0.0
        surrogate_loss = actor_loss if actor_loss is not None else 0.0
        mean_noise = noise_std if noise_std is not None else self.alg.actor_critic.std.mean().item()
        mean_reward = statistics.mean(locs['rewbuffer']) if len(locs['rewbuffer']) > 0 else 0.0
        mean_len = statistics.mean(locs['lenbuffer']) if len(locs['lenbuffer']) > 0 else 0.0

        # TensorBoard 写入
        if self.writer:
            self.writer.add_scalar('Loss/value_function', value_loss, locs['it'])
            self.writer.add_scalar('Loss/surrogate', surrogate_loss, locs['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_noise, locs['it'])
            self.writer.add_scalar('Perf/fps', fps, locs['it'])
            self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
            self.writer.add_scalar('Perf/learn_time', locs['learn_time'], locs['it'])
            if len(locs['rewbuffer']) > 0:
                self.writer.add_scalar('Train/mean_reward', mean_reward, locs['it'])
                self.writer.add_scalar('Train/mean_episode_length', mean_len, locs['it'])
                self.writer.add_scalar('Train/mean_reward/time', mean_reward, self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', mean_len, self.tot_time)


    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
