import pickle
from robot.data import RoboDemoDset
import matplotlib.pyplot as plt
import numpy as np
from reward_extraction.reward_functions import RobotLearnedRewardFunction
from r3m import load_r3m
import gym
from gym.wrappers.time_limit import TimeLimit
from rl.wrappers import wrap_gym
from robot.xarm_env import SimpleRealXArmReach, LrfRealXarmReach
from ml_collections import config_flags
from rl.agents import SACLearner
from absl import app, flags
from pathlib import Path




print("loading the environment")
r3m_net = load_r3m("resnet18")
r3m_net.to("cuda")
r3m_net.eval()
r3m_embedding_dim = 512
env = LrfRealXarmReach(
            control_frequency_hz = 8,
            scale_factor = 5,
            use_gripper = False,
            use_camera = True,
            use_r3m = True,
            r3m_net = r3m_net,
            random_reset_home_pose = False,
            low_collision_sensitivity = True,
            goal=np.array([45.4,-17.3, 18.0]), 
        )

env = wrap_gym(env, rescale_actions=True, obs_key="r3m_with_ppc")

env = TimeLimit(env, 40)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)


print("loading positive data ...")
demo_dir = "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/push"
expert_data = RoboDemoDset(f"{demo_dir}/demos.hdf", read_only_if_exists=True)

print("loading negative data")
on_policy_dir = "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/negative_trajs.pkl"
with open(on_policy_dir, "rb") as f:
    replay_buffer = pickle.load(f)

print("loading negative image data")
on_policy_img_dir = "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/negative_trajs_img.pkl"
with open(on_policy_img_dir, "rb") as f:
    img_replay_buffer = pickle.load(f)

print("loading lrf...")
repo_root = Path.cwd()
exp_dir = exp_dir = f'{repo_root}/walk_in_the_park/saved/gail_debug'
lrf = RobotLearnedRewardFunction(
        obs_size=r3m_embedding_dim,
        exp_dir=exp_dir,
        demo_path=f"{demo_dir}/demos.hdf",
        replay_buffer=replay_buffer,
        image_replay_buffer=img_replay_buffer,
        horizon=40,
        r3m_net=r3m_net,
        add_state_noise=False,
        train_classify_with_mixup=False,
        obs_is_image=False
    )
env.set_lrf(lrf)

lrf.train(500)

rgb_expert = []
eef_poses_expert = []
r3m_vecs_expert = []

rgb_all_expert = []
eef_poses_expert = []
r3m_vec_expert = []
for i in range(len(expert_data)):
    dat_rgb = expert_data.__getitem__(i)['rgb']
    dat_eef = expert_data.__getitem__(i)['eef_pose']
    dat_r3mvec = expert_data.__getitem__(i)['r3m_vec']
    rgb_all_expert.append(dat_rgb)
    eef_poses_expert.append(dat_eef)
    r3m_vec_expert.append(dat_r3mvec)

d = replay_buffer.dataset_dict
obs_orig = d['observations']
eef_pos_nonexpert = obs_orig[:, -3:]
eef_pos_nonexpert = eef_pos_nonexpert[:1000]
r3m_vec_nonexpert = obs_orig[:, :-3]
r3m_vec_nonexpert = r3m_vec_nonexpert[:1000]

r3m_vec_expert = np.asarray(r3m_vec_expert).reshape(-1,512)
eef_poses_expert = np.asarray(eef_poses_expert).reshape(-1, 3)

rewards_nonexpert = env.debug_reward(r3m_vec_nonexpert)[1]
rewards_expert = env.debug_reward(r3m_vec_expert)[1]

def vis():
    plt.subplot(1, 2, 1)
    plt.scatter3D(eef_pos_nonexpert[:-3, 0], eef_pos_nonexpert[:-3, 1], eef_pos_nonexpert[:-3, 2], c=rewards_nonexpert)
    plt.scatter3D(eef_poses_expert[:-3, 0], eef_poses_expert[:-3, 1], eef_poses_expert[:-3, 2], c=rewards_expert)
    plt.colorbar()
    plt.xlim(20, 70)
    plt.ylim(-10, 10)
    plt.subplot(1, 2, 2)
    plt.scatter3D(eef_poses_expert[:, 0], eef_poses_expert[:, 1], eef_poses_expert[:, 1])
    plt.xlim(20, 70)
    plt.ylim(-10, 10)
    plt.show()

def vis_3d():
    f1 = plt.figure(1)
    ax = plt.axes(projection ="3d")
    scttr1 = ax.scatter3D(eef_pos_nonexpert[:, 0], eef_pos_nonexpert[:, 1], eef_pos_nonexpert[:, 2], c=rewards_nonexpert)
    ax.set_xlim(20, 70)
    ax.set_ylim(-10, 10)
    ax.set_zlim(17, 43)
    f1.colorbar(scttr1, ax = ax)
    f2 = plt.figure(2)
    ax2 = plt.axes(projection = "3d")
    ax2.set_xlim(20, 70)
    ax2.set_ylim(-10, 10)
    ax2.set_zlim(17, 43)
    scttr2 = ax2.scatter3D(eef_poses_expert[:, 0], eef_poses_expert[:, 1], eef_poses_expert[:, 2], c=rewards_expert)
    f2.colorbar(scttr2, ax=ax2)
    f3 = plt.figure(3)
    ax3 = plt.axes(projection = "3d")
    ax3.set_xlim(20, 70)
    ax3.set_ylim(-10, 10)
    ax3.set_zlim(17, 43)
    ax3.scatter3D(eef_pos_nonexpert[:, 0], eef_pos_nonexpert[:, 1], eef_pos_nonexpert[:, 2], c='r')
    ax3.scatter3D(eef_poses_expert[:, 0], eef_poses_expert[:, 1], eef_poses_expert[:, 2], c='b')
    plt.show()








import IPython;
IPython.embed()

vis()
