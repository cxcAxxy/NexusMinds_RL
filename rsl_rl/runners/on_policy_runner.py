# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]#读取训练配置
        self.alg_cfg = train_cfg["algorithm"]#读取算法配置
        self.policy_cfg = train_cfg["policy"]#读取策略配置
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:#給critic选择观测维度
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)#实例化策略网络
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)#实例化PPO算法
        self.num_steps_per_env = self.cfg["num_steps_per_env"]#每个环境中每次采集的步数
        self.save_interval = self.cfg["save_interval"]#模型保存间隔

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])#初始化存储

        # Log
        self.log_dir = log_dir#日志保存路径
        self.writer = None# TensorBoard写入器
        self.tot_timesteps = 0#累计时间步数
        self.tot_time = 0#累计训练时间
        self.current_learning_iteration = 0#当前学习迭代次数

        _, _ = self.env.reset()#重置环境
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:#随机初始化回合长度，避免全部环境同时重置
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()#获取当前观测值
        privileged_obs = self.env.get_privileged_observations()#获取当前特权观测值
        critic_obs = privileged_obs if privileged_obs is not None else obs#选择critic的观测值
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)#移动到指定设备
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []#存储回合信息
        rewbuffer = deque(maxlen=100)#存储最近100个回合的奖励
        lenbuffer = deque(maxlen=100)#存储最近100个回合的长度
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)#当前回合奖励和
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)#当前回合长度

        tot_iter = self.current_learning_iteration + num_learning_iterations#总迭代次数
        for it in range(self.current_learning_iteration, tot_iter):#主训练循环
            start = time.time()
            # Rollout
            with torch.inference_mode():#不进行梯度计算
                for i in range(self.num_steps_per_env):#在每个环境的中采集指定步数的数据
                    actions = self.alg.act(obs, critic_obs)#选择动作
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)#获得执行动作后的环境反馈
                    critic_obs = privileged_obs if privileged_obs is not None else obs#选择critic的观测值
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)#移动到指定设备
                    self.alg.process_env_step(rewards, dones, infos)#处理环境步骤的结果
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])#记录回合信息
                        cur_reward_sum += rewards#累计当前回合奖励
                        cur_episode_length += 1#累计当前回合长度
                        new_ids = (dones > 0).nonzero(as_tuple=False)#找出哪些环境在本步结束了回合
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())#记录结束回合的奖励
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())#记录结束回合的长度
                        cur_reward_sum[new_ids] = 0#重置结束回合的奖励和
                        cur_episode_length[new_ids] = 0#重置结束回合的长度

                stop = time.time()
                collection_time = stop - start#数据采集时间

                # Learning step
                start = stop #继续计时
                self.alg.compute_returns(critic_obs)#计算回报
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()#更新策略与价值网络
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:#日志记录
                self.log(locals())
            if it % self.save_interval == 0:#模型保存
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()#清空回合信息缓存
        
        self.current_learning_iteration += num_learning_iterations#更新当前学习迭代次数
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))#保存最终模型

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs#计算并累加并次迭代的总交互数
        self.tot_time += locs['collection_time'] + locs['learn_time']#累加本次迭代的总时间
        iteration_time = locs['collection_time'] + locs['learn_time']#本次迭代时间

        ep_string = f''#空字符串初始化
        if locs['ep_infos']:#是否有回合信息
            for key in locs['ep_infos'][0]:#遍历回合信息的键
                infotensor = torch.tensor([], device=self.device)#初始化信息张量
                for ep_info in locs['ep_infos']:#遍历所有回合信息
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):#如果回合信息不是张量
                        ep_info[key] = torch.Tensor([ep_info[key]])#转换为张量
                    if len(ep_info[key].shape) == 0:#如果张量是零维的
                        ep_info[key] = ep_info[key].unsqueeze(0)#扩展为一维张量
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))#拼接张量
                value = torch.mean(infotensor)#计算信息的均值
                self.writer.add_scalar('Episode/' + key, value, locs['it'])#记录到TensorBoard
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""#添加回合信息字符串
        mean_std = self.alg.actor_critic.std.mean()#策略噪声标准差
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))#计算每秒步数

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])#记录价值函数损失
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])#记录代理损失
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])#记录学习率
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])#记录策略噪声标准差
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])#记录每秒步数
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])#记录数据采集时间
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])#记录学习时间
        if len(locs['rewbuffer']) > 0:#如果有奖励数据
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])#记录平均奖励
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])#记录平均回合长度
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)#记录累计时间的平均奖励
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)#记录累计时间的平均回合长度

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "#格式化迭代信息字符串

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):#保存模型
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),#保存模型参数
            'optimizer_state_dict': self.alg.optimizer.state_dict(),#保存优化器参数
            'iter': self.current_learning_iteration,#保存当前迭代次数
            'infos': infos,#保存额外信息
            }, path)

    def load(self, path, load_optimizer=True):#加载模型
        loaded_dict = torch.load(path)#加载保存的字典
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])#加载模型参数
        if load_optimizer:#是否加载优化器参数
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])#加载优化器参数
        self.current_learning_iteration = loaded_dict['iter']#加载当前迭代次数
        return loaded_dict['infos']#返回额外信息

    def get_inference_policy(self, device=None):#获取推理策略
        self.alg.actor_critic.eval() #设置神经网络为评估模式
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
