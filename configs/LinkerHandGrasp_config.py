from isaacgym import gymutil
import torch

args = gymutil.parse_arguments(
    description="test Gym Simulation",
    custom_parameters=[
        {"name": "--use_gpu", "type": bool, "default": True, "help": "Use GPU for physics"},
        {"name": "--use_gpu_pipeline", "type": bool, "default": True, "help": "Use GPU pipeline"},
        {"name": "--headless", "type": bool, "default": False, "help": "Run simulation without viewer"},
    ]
)


class GlobalCfg:
    """全局共享配置 - 统一管理所有组件的共同参数"""

    def __init__(self):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 环境数量
        self.num_envs = 4


class GymCfg:
    """仿真器配置"""

    def __init__(self, args=None):
        # 默认值
        self.headless = False
        self.use_gpu = True
        self.use_gpu_pipeline = True

        # 如果传了 args，就覆盖默认值
        if args is not None:
            for key, value in vars(args).items():
                setattr(self, key, value)


class RobotCfg:
    """机械臂配置"""

    def __init__(self, global_cfg):
        # 控制相关参数
        self.control_type = "ee"
        self.block_gripper = True
        self.num_actions = 14
        self.num_obs = 51
        self.num_envs = global_cfg.num_envs  # 修改其他配置一致
        self.control_type_sim = "effort"

        # 模型路径与姿态
        self.asset = "/home/cxc/Desktop/NexusMInds_RL/env/assets"
        self.robot_files = "urdf/frankaLinkerHand_description/robots/frankaLinker.urdf"
        # 每个机器人的初始位置是一样的吗
        self.base_pose = [0, 0, 0]  # 每个环境的机器人位置
        self.base_orn = [0, 0, 0, 1]  # 每个环境的机器人姿态

        #self.ee_link = "panda_hand"
        self.headless = "False"
        self.control_decimation = 6
        self.action_low = -1
        self.action_high = 1


class TaskCfg:
    """Franka Reach 任务配置"""

    def __init__(self, global_cfg):
        self.name = "Reach"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = global_cfg.num_envs  # 修改其他配置一致

        self.reward_type = "dense"
        self.distance_threshold = 0.02

        # 定义所有的参数，后续根据那公式划分一下公式
        self.c1 = 1
        self.c2 = 1
        self.c3 = 1
        self.c4 = 1
        self.c5 = 1
        self.c6 = 1

        self.alpha_mid =1
        self.alpha_pos =1


        # 改为字典的方式：
        self.reward_scales = {
            "grasp_goal_distance" : self.c1 * self.c4 * self.c5,
            "grasp_mid_point" : self.c1 * self.c4 * self.c6,
            "pos_reach_distance" : self.c2
        }


class AllCfg:
    """环境总体配置"""

    def __init__(self, global_cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = global_cfg.num_envs
        self.num_achieved_goal = 3
        self.num_desired_goal = 3
        self.max_episode_length = 200
        self.max_episode_length_s = 4.0  # 秒数形式（用于日志统计）
        self.decimation = 4
        self.control_type_sim = "effort"


class LinkGraspCfg:
    """总配置类"""

    def __init__(self):
        self.global_cfg = GlobalCfg()
        self.gymcfg = GymCfg(args)
        self.robotcfg = RobotCfg(self.global_cfg)
        self.taskcfg = TaskCfg(self.global_cfg)
        self.all = AllCfg(self.global_cfg)