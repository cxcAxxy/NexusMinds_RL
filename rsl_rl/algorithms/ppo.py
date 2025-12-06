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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,#每次更新时使用相同的数据集进行多轮训练
                 num_mini_batches=1,#将经验数据分成多少个小批量进行训练
                 clip_param=0.2,#PPO核心的裁剪参数
                 gamma=0.998,#折扣因子
                 lam=0.95,#GAE的平滑参数
                 value_loss_coef=1.0,#价值函数损失的权重
                 entropy_coef=0.0,#熵奖励的权重
                 learning_rate=1e-3,#优化器的学习率
                 max_grad_norm=1.0,#梯度裁剪的最大范数
                 use_clipped_value_loss=True,#是否使用裁剪的价值函数损失
                 schedule="fixed",#学习率调度方式
                 desired_kl=0.01,#期望的KL散度值
                 device='cpu',#使用的计算设备
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)# Adam优化器
        self.transition = RolloutStorage.Transition()# 用于存储单个时间步的过渡数据

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):#初始化存储
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):#设置为测试模式？
        self.actor_critic.test()
    
    def train_mode(self):#设置为训练模式？
        self.actor_critic.train()

    def act(self, obs, critic_obs):#根据当前观察值选择动作
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()#执行动作
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()#评估状态价值
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()#计算动作的对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach()#计算动作的均值
        self.transition.action_sigma = self.actor_critic.action_std.detach()#计算动作的标准差
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs#记录当前的观察值
        self.transition.critic_observations = critic_obs#记录当前的价值观察值
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):#处理环境步骤的结果
        self.transition.rewards = rewards.clone()#记录奖励
        self.transition.dones = dones#记录是否结束
        # Bootstrapping on time outs
        if 'time_outs' in infos:#处理时间超时的情况
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)#将过渡数据添加到存储中
        self.transition.clear()#清除过渡数据
        self.actor_critic.reset(dones)#重置智能体状态
    
    def compute_returns(self, last_critic_obs):#计算回报
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()#评估最后一个观察值的状态价值
        self.storage.compute_returns(last_values, self.gamma, self.lam)#计算回报和优势函数

    def update(self):
        mean_value_loss = 0#初始化平均价值损失
        mean_surrogate_loss = 0#初始化平均代理损失
        if self.actor_critic.is_recurrent:#根据是否为递归神经网络选择生成器？
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)#递归小批量生成器
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)#非递归小批量生成器
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:#遍历小批量数据

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])#生成动作分布
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)#计算动作的对数概率
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])#评估状态价值
                mu_batch = self.actor_critic.action_mean#获取动作均值
                sigma_batch = self.actor_critic.action_std#获取动作标准差
                entropy_batch = self.actor_critic.entropy#计算熵

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':#自适应调整学习率
                    with torch.inference_mode():#不进行梯度计算
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)#计算KL散度
                        kl_mean = torch.mean(kl)#计算平均KL散度

                        if kl_mean > self.desired_kl * 2.0:#根据KL散度调整学习率
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:#更新优化器的学习率
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))#计算新旧概率比率
                surrogate = -torch.squeeze(advantages_batch) * ratio#计算未裁减代理损失
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)#计算裁减后的代理损失
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()#选择接近0的代理损失作为最终损失

                # Value function loss
                if self.use_clipped_value_loss:#是否使用裁剪的价值函数损失
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)#裁剪后的状态价值
                    value_losses = (value_batch - returns_batch).pow(2)#计算未裁剪的价值损失
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)#计算裁剪后的价值损失
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()#选择接近0的价值损失作为最终损失
                else:#不使用裁剪的价值函数损失
                    value_loss = (returns_batch - value_batch).pow(2).mean()#计算价值损失

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()#总损失函数

                # Gradient step
                self.optimizer.zero_grad()#清除梯度
                loss.backward()#反向传播计算梯度
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)#梯度裁剪，防止梯度爆炸
                self.optimizer.step()#更新参数

                mean_value_loss += value_loss.item()#累积价值损失
                mean_surrogate_loss += surrogate_loss.item()#累积代理损失

        num_updates = self.num_learning_epochs * self.num_mini_batches#计算总的更新次数
        mean_value_loss /= num_updates#计算平均价值损失
        mean_surrogate_loss /= num_updates#计算平均代理损失
        self.storage.clear()#清除存储

        return mean_value_loss, mean_surrogate_loss
