import os
import pickle
from flax.training import checkpoints
from pathlib import Path
import gym
from matplotlib import pyplot as plt
from ml_collections import config_flags
from absl import app, flags
from rl.agents import SACLearner
from rl.wrappers import wrap_gym
from rl.data import ReplayBuffer
from rl.data.image_buffer import RAMImageReplayBuffer
from reward_extraction.reward_functions import RobotLearnedRewardFunction
from gym.wrappers.time_limit import TimeLimit
import tqdm
from cam.utils import VideoRecorder
from robot.xarm_env import LrfRealXarmReach
from robot.utils import HZ
from r3m import load_r3m
import numpy as np

GAIL=False
REG=False
NUM_EVAL=1
RSR_STR = 'drawer_open_nat_sub'
EXP_STR = 'drawer_open_nat_sub'
MAX_EPISODE_TIME_S = 4
use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"


FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 42, 'Random seed.')
# demo_dir does not matter
flags.DEFINE_string('demo_dir', "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/paper_reach", 'Demo directory')
config_flags.DEFINE_config_file(
    'config',
    'walk_in_the_park/configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)
repo_root = Path.cwd()
exp_dir = f'{repo_root}/walk_in_the_park/saved/{EXP_STR}'
chkpts_dir = f'{exp_dir}/checkpoints'
buffer_dir = f'{exp_dir}/buffers'
res_net_sz = 18
r3m_net = load_r3m(f"resnet{res_net_sz}")
r3m_net.to("cuda")
r3m_net.eval()
r3m_embedding_dim = (2048 if res_net_sz == 50 else 512)
rsr_dir = f'{repo_root}/walk_in_the_park/saved/rsr_{RSR_STR}'
max_steps = MAX_EPISODE_TIME_S * HZ

env = LrfRealXarmReach(
            control_frequency_hz = HZ,
            scale_factor = 5,
            use_gripper = use_gripper,
            use_camera = use_camera,
            use_r3m = use_r3m,
            r3m_net = r3m_net,
            random_reset_home_pose = False,
            low_collision_sensitivity = True,
            goal= [50.5, 9.84, 18.5], # for reach_nat [61.4, 20.0, 25.8], # for obst push: np.array([50, 16.4, 18]), 34, -22, 18.6
            obs_sz = r3m_embedding_dim,
            wait_on_reset=True
        )
env = wrap_gym(env, rescale_actions=True, obs_key=obs_key)
env = TimeLimit(env, max_steps)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

def get_chkpt_dirs():
    '''
    @Returns: an tuple of two arrays, the first array is the checkpoint directories
    and the second array is the step numbers of the checkpoints
    '''
    chkpt_dirs = []
    chkpt_steps = []
    for sub_dir in Path(chkpts_dir).iterdir():
        if sub_dir.is_dir():
            # find the file that starts with "checkpoint_"
            for file in sub_dir.iterdir():
                if file.is_file() and file.name.startswith("checkpoint_"):
                    cur_step = int(file.name.split("_")[1])
                    if cur_step < 4000:
                        continue
                    chkpt_dirs.append(str(file))
                    chkpt_steps.append(int(file.name.split("_")[1]))
    return chkpt_dirs, chkpt_steps

def resurrect_agents(chkpt_dirs):
    '''
    @Params chkpt_dirs: an array of strings that are the names of the checkpoints
    @Returns: an array of agents, the ith agent corresponds to the 
    ith eval checkpoint
    '''
    agents = []
    for chkpt_dir in chkpt_dirs:
        kwargs = dict(FLAGS.config)
        agent = SACLearner.create(FLAGS.seed, env.observation_space,
                                env.action_space, **kwargs)
        agent = checkpoints.restore_checkpoint(chkpt_dir, agent, parallel=False)
        agents.append(agent)
    return agents

def resurrect_lrfs(chkpt_dirs):
    '''
    @Params chkpt_dirs: an array of strings that are the names of the checkpoints
    @Returns: an array of lrf objects, the ith LRF object corresponds to the 
    ith eval checkpoint
    '''
    lrfs = []
    for chkpt_dir in chkpt_dirs:
        start_i = 0
        # here, we reinitialize both the ReplayBuffer and RAMImageReplayBuffer
        # because we do not use it to train the LRF
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     max_steps)
        # Our computer does not have enough ram for 50k
        img_replay_buffer = RAMImageReplayBuffer(capacity=5_000, img_shape=env.image_space.shape)
        replay_buffer.seed(FLAGS.seed)

        lrf = RobotLearnedRewardFunction(
            obs_size=r3m_embedding_dim,
            exp_dir=exp_dir,
            demo_path=f"{FLAGS.demo_dir}/demos.hdf",
            replay_buffer=replay_buffer,
            image_replay_buffer=img_replay_buffer,
            horizon= 4 * 8,
            add_state_noise=False,
            train_classify_with_mixup=False,
            obs_is_image=False,
            mask_reg=REG,
            log_rwd=False,
            plotting_mode=True,
            disable_ranking=GAIL
        )
        load_dir = chkpt_dir.split("/checkpoint_")[0]
        lrf.load_models(load_str=load_dir)
        lrfs.append(lrf)
    return lrfs



def eval(env, agent, lrf, cur_step, num_episodes: int = NUM_EVAL):
    '''
    Runs num_episodes of evaluation on the agent in the environment
    @Params 
        env: the environment
        agent: the agent
        cur_step: the step number of the agent
        num_episodes: the number of episodes to evaluate the agent
    '''
    print(f"Running {num_episodes} episodes of evaluation...")
    cur_step_str = str(cur_step).zfill(6)
    cur_eval_dir = Path(f"{rsr_dir}/{cur_step_str}")
    if not os.path.exists(str(cur_eval_dir)):
        os.makedirs(str(cur_eval_dir))
    
    recorder = VideoRecorder(save_dir=cur_eval_dir, fps=env.hz)

    env.set_lrf(lrf)


    all_progresses, all_masks, all_rewards = [], [], []
    for i in tqdm.tqdm(range(num_episodes)):
        observation, done = env.reset(), False
        rgbs, progresses, masks, rewards = [], [], [], []
        rgbs.append(env.rgb.copy())
        while not done:
            action = agent.eval_actions(observation)
            # print(f"action: {action}, eef xyz: {eefxyz_from_obs(observation)}")
            next_observation, r, done, info = env.step(action)
            progress, mask, reward = env.lrf.last_pmr()
            progresses.append(float(progress))
            masks.append(float(mask))
            rewards.append(float(reward))
            rgbs.append(env.rgb.copy())
            observation = next_observation
            if done:
                break

        # bookkeeping
        all_progresses.append(progresses)
        all_masks.append(masks)
        all_rewards.append(rewards)

        # save video
        for frame_idx, rgb_frame in enumerate(rgbs):
            if frame_idx == 0:
                recorder.init(rgb_frame)
            else:
                recorder.record(rgb_frame)
        save_str = str(i).zfill(2)
        recorder.save(f"{save_str}.mp4")

        gt_rwd = env.ground_truth_reward(observation)

        print(f"GROUND TRUTH REWARD: {gt_rwd}")

        # plot each video individually as well
        plt.clf(); plt.cla()
        plt.plot(progresses, label="progress")
        plt.plot(masks, label='mask')
        plt.plot(rewards, label='reward')
        traj_idx_str = str(i).zfill(2)
        plt.legend()
        plt.savefig(f"{cur_eval_dir}/{traj_idx_str}_pmr.png")

        # plot
        plt.clf(); plt.cla()
        for i, reward_traj in enumerate(all_rewards):
            plt.plot(reward_traj, label=f"{str(i).zfill(2)}_{np.sum(reward_traj)}")
        plt.legend()
        plt.savefig(f"{cur_eval_dir}/rewards.png")

        plt.clf(); plt.cla()
        for i, progress_traj in enumerate(all_progresses):
            plt.plot(progress_traj, label=f"{str(i).zfill(2)}")
        # plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f"{cur_eval_dir}/progresses.png")

        plt.clf(); plt.cla()
        for i, mask_traj in enumerate(all_masks):
            plt.plot(mask_traj, label=f"{str(i).zfill(2)}")
        # plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f"{cur_eval_dir}/masks.png")

def main(_):
    print("Resurrecting agents...")
    chkpt_dirs, chkpt_steps = get_chkpt_dirs()
    agents = resurrect_agents(chkpt_dirs)
    
    print("Recurrecting LRFs...")
    lrfs = resurrect_lrfs(chkpt_dirs)

    assert len(agents) == len(lrfs) == len(chkpt_steps)

    while True:
        try:
            print("\nPlease choose an option:")
            print("\t [1] Evaluate all agents")
            print("\t [2] Evaluate a specific agent")
            print("\t [3] Exit")

            option = int(input("Enter option: "))
            if option == 1:
                for i, (agent, lrf) in tqdm.tqdm(enumerate(zip(agents, lrfs))):
                    step_num = chkpt_steps[i]
                    print(f"\nEvaluating agent at step {step_num}:")
                    eval(env, agent, lrf, chkpt_steps[i])
            elif option == 2:
                ttl_agents_num = len(agents)
                for i in range(ttl_agents_num):
                    print(f"[{i}] Agent at step {chkpt_steps[i]}", end=" ")
                agent_idx = int(input(f"\nEnter agent index (there are a total of {ttl_agents_num} agents): "))
                eval(env, agents[agent_idx], lrfs[agent_idx],chkpt_steps[agent_idx])
            elif option == 3:
                break
            else:
                print("Invalid option, please try again.")
        except ValueError:
            print("Invalid option, please try again.")

    print("Exited. You can savely terminate this process now.")

if __name__ == "__main__":
    app.run(main)