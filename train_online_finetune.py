#! /usr/bin/env python
import os
from pathlib import Path
import pickle
import shutil

import torch

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import cv2 as cv
import torch
import torchvision.transforms as T
transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),  # divides by 255, will also convert to chw
            ])
from PIL import Image


import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.data.image_buffer import RAMImageReplayBuffer
from rl.data.util import *
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym
from r3m import load_r3m
from reward_extraction.models import Policy


from cam.utils import VideoRecorder

from robot.xarm_env import SimpleRealXArmReach, LrfRealXarmReach, FineTuneXArmReach

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('checkpoint_interval', 100, 'Checkpointing interval.') # 48
flags.DEFINE_integer('buffer_saving_interval', 2000, 'Buffer saving interval.') # 400
flags.DEFINE_integer('lrf_update_frequency', 1440, 'Update lrf every x timesteps.') # 240
flags.DEFINE_integer('eval_interval', 480, 'Eval interval.') # 128
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.') 
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1440),  # 240
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', True, 'Use real robot.')
flags.DEFINE_string('demo_dir', "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/push_fin", 'Demo directory')
flags.DEFINE_string('exp_str', 'push_fin', 'define experiment directory')
config_flags.DEFINE_config_file(
    'config',
    'walk_in_the_park/configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)



def eval(env, agent, exp_dir: str, curr_step: int, num_episodes: int = 3):
    print(f"running {num_episodes} eval episodes!")
    # run some eval episodes and plot the reward to
    # get a sense of what real data looks like
    curr_step_str = str(curr_step).zfill(6)
    video_dir = Path(f"{exp_dir}/eval/{curr_step_str}")
    if not os.path.exists(str(video_dir)):
        os.makedirs(str(video_dir))
    recorder = VideoRecorder(save_dir=video_dir, fps=env.hz)



    def eefxyz_from_obs(obs):
        if env.use_gripper:
            return obs[-15:][:3]
        else:
            return obs[-14:][:3]

    all_progresses, all_masks, all_rewards = [], [], []
    for i in tqdm.tqdm(range(num_episodes)):
        observation, done = env.reset(), False
        rgbs, progresses, masks, rewards = [], [], [], []
        rgbs.append(env.rgb.copy())
        while not done:
            action = agent.eval_actions(observation)
            # print(f"action: {action}, eef xyz: {eefxyz_from_obs(observation)}")
            next_observation, r, done, info = env.step(action)
            progress, mask, reward = r
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

        # plot each video individually as well
        plt.clf(); plt.cla()
        plt.plot(progresses, label="progress")
        plt.plot(masks, label='mask')
        plt.plot(rewards, label='reward')
        traj_idx_str = str(i).zfill(2)
        plt.legend()
        plt.savefig(f"{video_dir}/{traj_idx_str}_pmr.png")

        plt.clf(); plt.cla()
        plt.plot(np.exp(progresses), label="progress")
        plt.plot(np.exp(masks), label='mask')
        plt.plot(np.exp(rewards), label='reward')
        traj_idx_str = str(i).zfill(2)
        plt.legend()
        plt.savefig(f"{video_dir}/{traj_idx_str}_pmr_no_log.png")

    # plot
    plt.clf(); plt.cla()
    for i, reward_traj in enumerate(all_rewards):
        plt.plot(reward_traj, label=f"{str(i).zfill(2)}_{np.sum(reward_traj)}")
    plt.ylim(-3, 3)
    plt.legend()
    plt.savefig(f"{video_dir}/rewards.png")

    plt.clf(); plt.cla()
    for i, progress_traj in enumerate(all_progresses):
        plt.plot(progress_traj, label=f"{str(i).zfill(2)}")
    plt.ylim(-3, 3)
    plt.legend()
    plt.savefig(f"{video_dir}/progresses.png")

    plt.clf(); plt.cla()
    for i, mask_traj in enumerate(all_masks):
        plt.plot(mask_traj, label=f"{str(i).zfill(2)}")
    plt.ylim(-3, 3)
    plt.legend()
    plt.savefig(f"{video_dir}/masks.png")

    plt.clf(); plt.cla()
    for i, reward_traj in enumerate(all_rewards):
        plt.plot(np.exp(reward_traj), label=f"{str(i).zfill(2)}_{np.sum(reward_traj)}")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{video_dir}/rewards_nolog.png")

    plt.clf(); plt.cla()
    for i, progress_traj in enumerate(all_progresses):
        plt.plot(np.exp(progress_traj), label=f"{str(i).zfill(2)}")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{video_dir}/progresses_nolog.png")

    plt.clf(); plt.cla()
    for i, mask_traj in enumerate(all_masks):
        plt.plot(np.exp(mask_traj), label=f"{str(i).zfill(2)}")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{video_dir}/masks_nolog.png")



def main(_):
    wandb.init(project='real_franka_reach')
    wandb.config.update(FLAGS)

    from gym.wrappers.time_limit import TimeLimit
    from reward_extraction.reward_functions import RobotLearnedRewardFunction
    # from robot.franka_env import SimpleRealFrankReach, LrfRealFrankaReach, LrfCabinetDoorOpenFranka
    from robot.utils import HZ

    # defaults
    use_gripper, use_camera, use_r3m, obs_key = False, False, False, None
    MAX_EPISODE_TIME_S = 6 # 16
    MAX_STEPS = HZ * MAX_EPISODE_TIME_S
    DEMO_HRZ = 4 # 8

    # exp_str = 'r3m50_experiment'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_vec"
    exp_str = FLAGS.exp_str; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"
    # exp_str = "codetest"; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"

    repo_root = Path.cwd()
    exp_dir = f'{repo_root}/walk_in_the_park/saved/{exp_str}'

    # load r3m here so multiple things can use it without having multiple resnets loaded into GPU memory
    # r3m_net = load_r3m("resnet50")
    res_net_sz = 18
    r3m_net = load_r3m(f"resnet{res_net_sz}")
    r3m_net.to("cuda")
    r3m_net.eval()
    r3m_embedding_dim = (2048 if res_net_sz == 50 else 512) #512 #2048

    if FLAGS.real_robot:
        env = FineTuneXArmReach(
            control_frequency_hz = HZ,
            scale_factor = 20,
            use_gripper = use_gripper,
            use_camera = use_camera,
            use_r3m = use_r3m,
            r3m_net = r3m_net,
            random_reset_home_pose = False,
            low_collision_sensitivity = True,
            goal=np.load("/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/R2R_data_end.npy"),
            obs_sz = r3m_embedding_dim,
            wait_on_reset=True
        )
    else:
        assert False # what are you doing

    env = wrap_gym(env, rescale_actions=True, obs_key=obs_key)

    env = TimeLimit(env, MAX_STEPS)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)


    chkpt_dir = f'{exp_dir}/checkpoints'
    # commented back in
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = f'{exp_dir}/buffers'
    os.makedirs(buffer_dir, exist_ok=True)
    img_buffer_path = f'{buffer_dir}/image_buffer.hdf'

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        # Our computer does not have enough ram for 50k
        img_replay_buffer = RAMImageReplayBuffer(capacity=5_000, img_shape=env.image_space.shape)
        replay_buffer.seed(FLAGS.seed)
        print(f"no checkpoint!")
    else:
        start_i = int(last_checkpoint.split('_')[-1])

        agent = checkpoints.restore_checkpoint(last_checkpoint, agent, parallel=False)

        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)
        img_replay_buffer.restore_from_disk(buffer_dir)

        print(f"restoring checkpoint! {last_checkpoint} at t: {start_i}")

    last_eval_idx = start_i
    last_chkpt_idx = start_i
    last_buffer_save = start_i


    '''
    setup learned reward function
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_sz = 1024
    hidden_depth = 6
    hidden_layer_sz = 4096
    classify_net = Policy(obs_sz, 1, hidden_layer_sz, hidden_depth).to(device)
    classify_net.to(device)
    classify_net.load_state_dict(torch.load(f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/classify_net.pt"))

    classify_net.eval()

    ranking_net = Policy(obs_sz, 1, hidden_layer_sz, hidden_depth).to(device)
    ranking_net.to(device)
    ranking_net.load_state_dict(torch.load(f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/ranking_net.pt"))

    ranking_net.eval()

    env.set_net(ranking_net, classify_net)
    


    '''
    train video recorder to see how live data is being evaluated
    '''
    train_video_dir = f"{exp_dir}/train"
    train_video_path = Path(train_video_dir)
    if not os.path.exists(str(train_video_dir)):
        os.makedirs(str(train_video_dir))
    train_recorder = VideoRecorder(save_dir=train_video_path, fps=env.hz)
    progresses, masks, rewards = [], [], []

    observation, done = env.reset(), False
    image = env.rgb

    train_recorder.init(image)
    rewards = []

    # from rl.data.util import add_to_ds
    # add_to_ds(env, replay_buffer, f"{FLAGS.demo_dir}/demos.hdf")

    '''
    directory for gail debugging
    '''
    expert_plt_dir = f"{exp_dir}/demo_plt"
    expert_plt_path = Path(expert_plt_dir)
    if not os.path.exists(str(expert_plt_dir)):
        os.makedirs(str(expert_plt_dir))
    

    observations, actions, dones, next_observations = [], [], [], [] 

    goal=np.load("/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/R2R_data_end.npy"),

    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        next_image = env.rgb
        reward = env.calculate_reward(image, VideoRecorder(save_dir=train_video_path, fps=env.hz), "temp.mp4")
        progress, mask, reward = reward
        train_recorder.record(next_image)
        progresses.append(float(progress))
        masks.append(float(mask))
        rewards.append(float(reward))
    

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        img_replay_buffer.add(image)
        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation
        image = next_image

        if done:
            '''
            video recorder + debug plot
            '''
            save_str = str(i).zfill(7)
            train_recorder.save(f"{save_str}.mp4")

            train_save_path = f"{train_video_dir}/{save_str}.mp4"

            cap = cv.VideoCapture(str(train_save_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    pil_img = Image.fromarray(frame)
                except:
                    continue
                frame = transform(pil_img).unsqueeze(0).cpu().numpy()
                frames.append(frame)
            frames = torch.tensor(np.array(frames))
            frames = frames.squeeze(1).to(device)
            # import IPython; IPython.embed()
            with torch.no_grad():
                frames = r3m_net(frames * 255.0)
            frames = frames.squeeze().cpu().numpy()

            plt.clf(); plt.cla()
            plt.plot(progresses, label="progress")
            plt.plot(masks, label='mask')
            plt.plot(rewards, label='reward')
            plt.legend()
            plt.savefig(f"{train_video_dir}/{save_str}_pmr.png")



            # from IPython import embed; 
            # embed()


            '''
            run some eval episodes if we haven't run one in a while
            '''
            if i - last_eval_idx > FLAGS.eval_interval and i >= FLAGS.start_training:
                eval(env, agent, exp_dir, i)
                last_eval_idx = i
                curr_step_str = str(i).zfill(6)
                savepath = f"{exp_dir}/eval/{curr_step_str}"
                checkpoints.save_checkpoint(savepath,
                                        agent,
                                        step=i,
                                        keep=20,
                                        overwrite=True)

            if i - last_chkpt_idx > FLAGS.checkpoint_interval and i >= FLAGS.start_training:
                print("Saving Checkpoint!")
                last_chkpt_idx = i
                curr_step_str = str(i).zfill(6)
                savepath = f"{exp_dir}/checkpoints/{curr_step_str}"
                checkpoints.save_checkpoint(savepath,
                                        agent,
                                        step=i,
                                        keep=20,
                                        overwrite=True)

            if i - last_buffer_save > FLAGS.buffer_saving_interval and i >= FLAGS.start_training:
                print("Saving Buffer!")
                last_buffer_save = i
                try:
                    shutil.rmtree(buffer_dir)
                except:
                    pass

                os.makedirs(buffer_dir, exist_ok=True)
                with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                    pickle.dump(replay_buffer, f)
                img_replay_buffer.save_to_disk(buffer_dir)

            '''
            proper reset of the environment and some logging
            '''
            observation, done = env.reset(), False
            #observation, done env.step(np.array([0,0,0]))
            image = env.rgb
   
            progresses, masks, rewards = [], [], []
            train_recorder.init(image)

            # for k, v in info['episode'].items():
            #     decode = {'r': 'return', 'l': 'length', 't': 'time'}
            #     wandb.log({f'training/{decode[k]}': v}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)

            # update RL then update agent
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)



def pause_till_enter():
    import keyboard
    val = input("PRESS 'ENTER' key to CONTINUE")
    is_enter = " "
    while not is_enter:
        val = input("PRESS 'ENTER' key to CONTINUE")
        is_enter = " "

if __name__ == '__main__':
    app.run(main)