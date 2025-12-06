import numpy as np
import copy
import torch
import torch.nn as nn

class DDPGActorCritic(nn.Module):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[400, 300],
        critic_hidden_dims=[400, 300],
        activation="relu",
        n_critics=2,
        init_noise_std=0.1,
    ):
        super().__init__()

        activation_fn = get_activation(activation)
        self.num_actions = num_actions
        self.n_critics = n_critics

        # ========== Actor ==========
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_layers.append(activation_fn)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        actor_layers.append(nn.Tanh())  # 输出范围 [-1, 1]
        self.actor = nn.Sequential(*actor_layers)

        # ========== Critics ==========
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            critic_layers = [
                nn.Linear(num_critic_obs + num_actions, critic_hidden_dims[0]),
                activation_fn,
            ]
            for l in range(len(critic_hidden_dims) - 1):
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
            critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
            self.q_networks.append(nn.Sequential(*critic_layers))

        # --- Target networks ---
        self.actor_target = copy.deepcopy(self.actor)
        self.q_target = copy.deepcopy(self.q_networks)

        # --- Action noise parameter ---
        self.noise_std = init_noise_std

        # --- Action range (to be set from env) ---
        self.action_low = -1.0
        self.action_high = 1.0

        print(f"Actor: {self.actor}")
        print(f"Critics: {self.q_networks}")

    # ================= Actor =================
    def act(self, obs):
        """Deterministic action"""
        return self.actor(obs)

    def act_with_noise(self, obs):
        """Action with Gaussian exploration noise"""
        mu = self.actor(obs)
        noise = torch.randn_like(mu) * self.noise_std
        return torch.clamp(mu + noise, -1.0, 1.0)

    # ================= Critics =================
    def q_values(self, obs, act):
        q_input = torch.cat([obs, act], dim=-1)
        return [q(q_input) for q in self.q_networks]

    def q1(self, obs, act):
        q_input = torch.cat([obs, act], dim=-1)
        return self.q_networks[0](q_input)

    # ================= Targets =================
    @torch.no_grad()
    def soft_update(self, tau=0.005):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * param.data)
        for qs, qts in zip(self.q_networks, self.q_target):
            for param, target_param in zip(qs.parameters(), qts.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

    def sample_random_action(self, batch_size=None):

        device = next(self.actor.parameters()).device  # 保证在 actor 所在设备

        action = torch.empty((batch_size, self.num_actions), device=device).uniform_(self.action_low, self.action_high)
        return action
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean


# Helper
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return nn.ReLU()
