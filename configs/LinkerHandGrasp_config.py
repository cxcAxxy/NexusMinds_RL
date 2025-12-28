from isaacgym import gymutil
import torch

args = gymutil.parse_arguments(
    description="test Gym Simulation",
    custom_parameters=[
        {"name": "--use_gpu", "type": bool, "default": True, "help": "Use GPU for physics"},
        {"name": "--use_gpu_pipeline", "type": bool, "default": False, "help": "Use GPU pipeline"},
        {"name": "--headless", "type": bool, "default": False, "help": "Run simulation without viewer"},
        {"name": "--logdir", "type": str, "default": "logs", "help": "Directory for logging"},
        {"name": "--num_envs", "type":int, "default":20 , "help": "the number of environments to train"},
    ]   
)

class GlobalCfg:
    """全局共享配置 - 统一管理所有组件的共同参数"""

    def __init__(self,args):
        # 设备配置
        self.device = args.sim_device if torch.cuda.is_available() and args.use_gpu else 'cpu'
        # 环境数量
        self.num_envs = args.num_envs
        self.logdir = args.logdir

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

    def __init__(self,global_cfg):
        # 控制相关参数
        self.control_type = "ee"
        self.block_gripper = True
        self.num_actions = 18
        self.num_obs = 51
        self.robot_num_dofs = 18
        self.num_envs = global_cfg.num_envs  # 修改其他配置一致
        self.control_type_sim = "effort"

        # 模型路径与姿态
        self.asset = "/home/gu/NexusMInds_RL/env/assets"
        self.robot_files = "urdf/frankaLinkerHand_description/robots/frankaLinker.urdf"
        # 每个机器人的初始位置是一样的吗
        self.base_pose = [0, 0, 0]  # 每个环境的机器人位置
        self.base_orn = [0, 0, 0, 1]  # 每个环境的机器人姿态

        #self.ee_link = "panda_hand"
        self.headless = "False"
        self.control_decimation = 6
        self.action_low = -1
        self.action_high = 1

        #120 22
        self.stiffness_robot = 120 #60
        self.damping_robot = 22 #8
        #40 10
        self.stiffness_hand = 30
        self.damping_hand = 10

        self.stiffness = {
            "panda_joint1": self.stiffness_robot,
            "panda_joint2": self.stiffness_robot,
            "panda_joint3": self.stiffness_robot,
            "panda_joint4": self.stiffness_robot,
            "panda_joint5": self.stiffness_robot, 
            "panda_joint6": self.stiffness_robot,
            "panda_joint7": self.stiffness_robot,
            "thumb_cmc_yaw" : self.stiffness_hand,
            "thumb_cmc_pitch" : self.stiffness_hand,
            "thumb_ip" : self.stiffness_hand,
            "index_mcp_pitch" : self.stiffness_hand,
            "index_dip" : self.stiffness_hand,
            "middle_mcp_pitch" : self.stiffness_hand,
            "middle_dip" : self.stiffness_hand,
            "ring_mcp_pitch" : self.stiffness_hand,
            "ring_dip" : self.stiffness_hand,
            "pinky_mcp_pitch" : self.stiffness_hand,
            "pinky_dip" : self.stiffness_hand
        }

        self.damping = {
            "panda_joint1": self.damping_robot,
            "panda_joint2": self.damping_robot,
            "panda_joint3": self.damping_robot,
            "panda_joint4": self.damping_robot,
            "panda_joint5": self.damping_robot,
            "panda_joint6": self.damping_robot,
            "panda_joint7": self.damping_robot,
            "thumb_cmc_yaw" : self.damping_hand,
            "thumb_cmc_pitch" : self.damping_hand,
            "thumb_ip" : self.damping_hand,
            "index_mcp_pitch" : self.damping_hand,
            "index_dip" : self.damping_hand,
            "middle_mcp_pitch" : self.damping_hand,
            "middle_dip" : self.damping_hand,
            "ring_mcp_pitch" : self.damping_hand,
            "ring_dip" : self.damping_hand,
            "pinky_mcp_pitch" : self.damping_hand,
            "pinky_dip" : self.damping_hand
        }


class TaskCfg:
    """Franka Reach 任务配置"""

    def __init__(self,global_cfg):
        self.name = "Reach"
        self.device = global_cfg.device
        self.num_envs = global_cfg.num_envs  # 修改其他配置一致

        self.reward_type = "dense"
        self.distance_threshold = 0.02

        # 定义所有的参数，后续根据那公式划分一下公式
        self.c1 = 1
        self.c2 = 3
        self.c3 = 1
        self.c4 = 1
        self.c5 = 2
        self.c6 = 3
        self.c7 = 1
        self.c8 = 1
        self.c9 = 1
        self.c10 = 2
        self.c11 = 5

        self.alpha_mid =1.5
        self.alpha_pos =1.5
        self.alpha_down = 1.5
        self.alpha_z = 6

        # 改为字典的方式：
        self.reward_scales = {
            "grasp_goal_distance" : self.c1 * self.c4 * self.c5,
            "grasp_mid_point" : self.c1 * self.c4 * self.c6,
            "pos_reach_distance" : self.c2 ,
            "finger_collision_reset": self.c7,
            "body_collision_reset": self.c8,
            "obj_reset": self.c9,
            "hand_down": self.c10,
            # "finger_z_distance": self.c11
        }


class AllCfg:
    """环境总体配置"""

    def __init__(self,global_cfg):
        self.device = global_cfg.device
        self.num_envs = global_cfg.num_envs
        self.num_achieved_goal = 3
        self.num_desired_goal = 3
        self.max_episode_length = 800
        self.max_episode_length_s = 4.0  # 秒数形式（用于日志统计）
        self.decimation = 4
        self.control_type_sim = "effort"


class LinkGraspCfg:
    """总配置类"""

    def __init__(self):
        self.global_cfg = GlobalCfg(args)
        self.gymcfg = GymCfg(args)
        self.robotcfg = RobotCfg(self.global_cfg)
        self.taskcfg = TaskCfg(self.global_cfg)
        self.all = AllCfg(self.global_cfg)