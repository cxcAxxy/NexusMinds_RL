import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rsl_rl.modules import DDPGActorCritic
class ReplayBuffer:
    def __init__(self, max_size, num_envs, obs_dim, act_dim, device):#经验回放缓冲区
        self.max_size = max_size# 最大容量
        self.num_envs = num_envs# 并行环境数量
        self.size = 0# 当前存储的样本数量
        self.device = device# 计算设备
        self.ptr = 0# 当前写入位置
        self.full = False# 是否已满

        self.obs_buf = torch.zeros((self.max_size, num_envs, obs_dim), dtype=torch.float32, device=self.device)# 状态缓冲区
        self.next_obs_buf = torch.zeros((self.max_size, num_envs, obs_dim), dtype=torch.float32, device=self.device)# 下一个状态缓冲区
        self.act_buf = torch.zeros((self.max_size, num_envs, act_dim), dtype=torch.float32, device=self.device)# 动作缓冲区
        self.rew_buf = torch.zeros((self.max_size, num_envs, 1), dtype=torch.float32, device=self.device)# 奖励缓冲区
        self.done_buf = torch.zeros((self.max_size, num_envs, 1), dtype=torch.float32, device=self.device)# 终止标志缓冲区
        
    def add(self, obs, act, rew, done, next_obs):#将数据添加到缓冲区

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        
        # 随机索引，torch 版本
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)#随机采样 batch_size 个索引

        # 取出对应的 transitions
        obs = self.obs_buf[idx]          # [batch_size, num_envs, obs_dim]
        act = self.act_buf[idx]          # [batch_size, num_envs, act_dim]
        rew = self.rew_buf[idx]          # [batch_size, num_envs, 1]
        next_obs = self.next_obs_buf[idx]# [batch_size, num_envs, obs_dim]
        done = self.done_buf[idx]        # [batch_size, num_envs, 1]

        # 如果希望展开 env 维度成标准训练 batch: [batch_size * num_envs, dim]
        obs = obs.reshape(-1, obs.shape[-1])
        act = act.reshape(-1, act.shape[-1])
        rew = rew.reshape(-1, 1)
        next_obs = next_obs.reshape(-1, next_obs.shape[-1])
        done = done.reshape(-1, 1)

        return obs, act, rew, next_obs, done

    def __len__(self):
        return self.size if self.full else self.ptr


# ================= DDPG/TD3-style Algorithm =================
class DDPG:
    """Off-policy TD3/DDPG-style algorithm using DDPGActorCritic"""
    actor_critic: DDPGActorCritic
    def __init__(self, actor_critic, device='cpu', gamma=0.99, tau=0.005, batch_size=256, lr=3e-4):
        self.ac = actor_critic
        self.device = device
        self.gamma = gamma
        self.tau = tau# 软更新系数
        self.batch_size = batch_size
        self.replay_buffer = None

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), lr=lr)#actor优化器
        self.critic_opt = torch.optim.Adam([p for q in self.ac.q_networks for p in q.parameters()], lr=lr)#critic优化器

    def init_storage(self, buffer_size, num_envs, obs_shape, act_shape):#初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(int(buffer_size), num_envs, obs_shape[0], act_shape[0], self.device)

    def store_transition(self, obs, act, rew, done, next_obs):#存储单个过渡
        self.replay_buffer.add(obs, act, rew, done, next_obs)

    def update(self):
        # 并行环境下可用样本总数
        total_samples = len(self.replay_buffer)
        if total_samples < self.batch_size:
            return None, None, None

        obs, act, rew, next_obs, done = self.replay_buffer.sample(self.batch_size)#采样一个批次的数据

        # Critic update
        with torch.no_grad():# 计算目标 Q 值(无梯度)
            next_act = self.ac.actor_target(next_obs)# 目标 actor 网络生成下一个动作
            target_q_list = [q(torch.cat([next_obs, next_act], dim=-1)) for q in self.ac.q_target]# 将下一个状态和动作拼接后对每个目标 Q 网络计算Q值
            target_q = torch.min(torch.cat(target_q_list, dim=1), dim=1, keepdim=True)[0]# 取最小的 Q 值作为目标 Q 值
            target = rew + self.gamma * (1 - done) * target_q# 计算目标 Q 值

        q_vals = self.ac.q_values(obs, act)#使用当前的 Q 网络计算当前 Q 值，返回多个 Q 值列表
        critic_loss = sum(F.mse_loss(q_val, target) for q_val in q_vals) / len(q_vals)# 计算均方误差损失
        self.critic_opt.zero_grad()# 清零梯度
        critic_loss.backward()# 反向传播
        self.critic_opt.step()# 优化器更新参数

        # Actor update
        actor_loss = -self.ac.q1(obs, self.ac.actor(obs)).mean()# 使用第一个q网络，取最大化 Q 值即最小化负 Q 值
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        self.ac.soft_update(self.tau)# 软更新目标网络

        mean_noise_std = getattr(self.ac, "noise_std", 0.0)# 计算当前噪声标准差的均值
        return critic_loss.item(), actor_loss.item(), mean_noise_std

