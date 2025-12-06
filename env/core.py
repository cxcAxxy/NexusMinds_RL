from abc import ABC, abstractmethod
# 这个是pytho内置模块abc，用来创建 抽象类和抽象方法,abstractmethod,用来标记子类必须实现的方法，子类如果没有实现这些方法，也不能实例化
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch



class Robot(ABC):
    """Base class for robot env.

    Args:
        sim (isaac gym or MuJoCo): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(self):
        pass

    # # 将神经网络输出 转换成物理仿真引擎能理解的控制命令
    # @abstractmethod
    # def set_action(self, action: np.ndarray) -> None:
    #     """Set the action. Must be called just before sim.step().
    #
    #     Args:
    #         action (np.ndarray): The action.
    #     """
    @abstractmethod
    def step(self,action) -> None:
        """
        Args:
            action:
        """


    @abstractmethod
    def get_obs(self) -> torch.Tensor:
        """Return the observation associated to the robot.

        Returns:
            torch.Tensor: The observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""


class Task(ABC):
    """Base class for tasks.
    Args:
        sim (issac gym or MuJuCo): Simulation instance.
    """

    def __init__(self, sim) -> None:
        self.sim = sim
        self.goal = None


    # 主要是 set a new goal,看任务需要重置吗，还是说一样。
    @abstractmethod
    def reset_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Reset the task: sample a new goal."""

    # return the observation associated to the task,这个后续比较有用，用于扩展。
    @abstractmethod
    def get_obs(self) -> torch.Tensor:
        """Return the observation associated to the task."""


    #  这个achieved_goal理解为 当前机器人的已经达到的目标状态。
    @abstractmethod
    def get_achieved_goal(self) -> torch.Tensor:
        """Return the achieved goal."""

    def get_goal(self) -> torch.Tensor:
        """Return the current desired goal."""
        if self.goal is None:
            raise RuntimeError("Goal not set yet — call reset() first.")
        return self.goal.clone()

    @abstractmethod
    def is_success(
        self,
        achieved_goal: torch.Tensor,
        desired_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(
        self,
        achieved_goal: torch.Tensor,
        desired_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv():

    def __init__(
        self,
        robot: Robot,
        task: Task,
        cfg: Any,

    ) -> None:

        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        # self.render_mode = self.sim.render_mode
        # self.metadata["render_fps"] = 1 / self.sim.dt
        self.robot = robot
        self.task = task
        self.cfg = cfg  # 保存配置对象
        # 优先使用模拟器的device，保持与Isaac Gym张量一致
        self.device = getattr(self.sim, 'device', cfg.all.device)
        self.num_envs=cfg.all.num_envs
        self.num_actions=self.robot.num_actions

        self.num_obs = self.robot.num_obs
        self.num_privileged_obs=None    # 后续更新
        self.num_achieved_goal = cfg.all.num_achieved_goal
        self.num_desired_goal = cfg.all.num_desired_goal


        self.max_episode_length=cfg.all.max_episode_length

        # allocate buffers
        self.obs_buf=torch.zeros(self.num_envs,self.num_obs,device=self.device,dtype=torch.float)
        self.achieved_goal_buf=torch.zeros(self.num_envs,self.num_achieved_goal,device=self.device,dtype=torch.float)
        self.desired_goal_buf=torch.zeros(self.num_envs,self.num_desired_goal,device=self.device,dtype=torch.float)

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,dtype=torch.float)
        else:
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.compute_reward_task=task.compute_reward
        self.extras = {}
        # 新增：按环境累计每回合奖励
        self.episode_sums = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def get_observations(self):
        """Get current observations for RSL-RL compatibility"""
        return self.obs_buf

    def get_privileged_observations(self):
        """Get privileged observations for RSL-RL compatibility"""
        return self.privileged_obs_buf

    def get_achieved_goal_obs(self):
        return self.achieved_goal_buf

    def get_desired_goal_obs(self):
        return self.desired_goal_buf

    #重置特定的环境
    def reset_idx(self,env_ids):
        if len(env_ids) == 0:
            return

        self.robot.reset_ids(env_ids)
        self.task.reset_ids(env_ids)

        # 在清零前，先把本回合的累计奖励写入日志信息
        self.extras["episode"] = {}
        self.extras["episode"]["goal_reward"] = torch.mean(self.episode_sums[env_ids]) / self.cfg.all.max_episode_length_s

        self.extras["time_outs"]=self.time_out_buf

        # 重置buffer的变量
        self.rew_buf[env_ids]=0.
        self.episode_length_buf[env_ids]=0.
        self.reset_buf[env_ids]=1.

        # 清空该回合累计
        self.episode_sums[env_ids] = 0.

        # send timeout info to the algorithm


    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        #重置的另一种写法，按照初始的状态来,还有一点就是，应该还有各种的 achieved_goal,还有对应的goal
        # obs, privileged_obs,achieved_goal,desired_goal, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs


    def step(self, action: torch.Tensor):
        # 修改为 多环境的，dones,还有extras，对应就是内部的信息。
        # 确保动作在与仿真相同的设备上

        if action.device != self.device:
            action = action.to(self.device)
        action_sim = self.robot.step(action)

        # 这个地方设置 control.decimation。

        for _ in range(self.cfg.all.decimation):
            self.sim.step(action_sim, self.cfg.all.control_type_sim)

        # 更新的问题！！！！，这个更新放到仿真环境里面，就是robot的接口一定要是完全合适的。
  

        self.post_physics_step()


        # 这个地方的逻辑有问题

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        #更新buf的数值。
        self.episode_length_buf += 1
        self.check_termination()

        # 顺序上的问题,注意一下
        self.update_observations()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

    # 更新对应buffer数值。


    def check_termination(self):
        """ Check if environments need to be reset
        """
        #self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        #self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        # 这个地方的仿真接口，就是 reset_buf的判断条件.

        #这个地方也不进行一个更新，判断条件后面再说，根据任务后续设定

        # 假设这个地方的判断：
        # 后续根据碰撞进行修改，或者是其他的逻辑判断
        self.reset_buf = 0

        collision_info = self.robot.check_ee_collision()
        collision_termination = collision_info['collision_occurred']

        task_success = self.task.is_success(
            self.task.get_achieved_goal(), 
            self.task.get_goal()
        )

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf = self.time_out_buf | collision_termination | task_success
    
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf=self.compute_reward_task(self.achieved_goal_buf,self.desired_goal_buf)
        # 累计到每回合和
        self.episode_sums += self.rew_buf

    def _reward_termination(self):#终止奖励
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf


    def update_observations(self):
        # 更新对应
        robot_buffer=self.robot.get_obs()
        task_buffer=self.task.get_obs()
        combined_buffer = torch.cat([robot_buffer, task_buffer], dim=1)
        self.obs_buf=combined_buffer
        self.desired_goal_buf=self.task.get_goal()
        self.achieved_goal_buf=self.task.get_achieved_goal()

    def close(self) -> None:
        self.sim.close()
