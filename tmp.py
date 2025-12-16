

# 首先导入包含 Isaac Gym 的配置和环境模块
from configs.LinkerHandGrasp_config import LinkGraspCfg
from env.TaskRobotEnv import LinkerHandGraspGym

# 然后导入可能包含 PyTorch 的 rsl_rl 模块
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO
from rsl_rl.utils import class_to_dict


def train():
    cfg = LinkGraspCfg()
    train_cfg = class_to_dict(rslCfgPPO())

    # 使用正确的环境类和配置
    env = LinkerHandGraspGym(cfg)

    while True:

        env.sim.gym.fetch_results(env.sim.sim, True)
        env.sim.gym.step_graphics(env.sim.sim)
        env.sim.gym.draw_viewer(env.sim.viewer, env.sim.sim, True)
        env.sim.gym.sync_frame_time(env.sim.sim)

if __name__ == '__main__':
    train()
