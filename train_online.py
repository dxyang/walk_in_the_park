#! /usr/bin/env python
import os
from pathlib import Path
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.data.image_buffer import RAMImageReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym
from r3m import load_r3m

from cam.utils import VideoRecorder

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('checkpoint_interval', 5000, 'Checkpointing interval.')
flags.DEFINE_integer('lrf_update_frequency', 1000, 'Update lrf every x timesteps.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
# flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
# flags.DEFINE_integer('action_history', 1, 'Action history.')
# flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', True, 'Use real robot.')
flags.DEFINE_string('demo_dir', "/home/dxy/code/rewardlearning-robot/data/demos/cabinet", 'Demo directory')
config_flags.DEFINE_config_file(
    'config',
    'walk_in_the_park/configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def eval(env, agent, exp_dir: str, curr_step: int, num_episodes: int = 5):
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

        # plot each video individually as well
        plt.clf(); plt.cla()
        plt.plot(progresses, label="progress")
        plt.plot(masks, label='mask')
        plt.plot(rewards, label='reward')
        plt.ylim(0, 1)
        traj_idx_str = str(i).zfill(2)
        plt.legend()
        plt.savefig(f"{video_dir}/{traj_idx_str}_pmr.png")

    # plot
    plt.clf(); plt.cla()
    for i, reward_traj in enumerate(all_rewards):
        plt.plot(reward_traj, label=f"{str(i).zfill(2)}_{np.sum(reward_traj)}")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{video_dir}/rewards.png")

    plt.clf(); plt.cla()
    for i, progress_traj in enumerate(all_progresses):
        plt.plot(progress_traj, label=f"{str(i).zfill(2)}")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{video_dir}/progresses.png")

    plt.clf(); plt.cla()
    for i, mask_traj in enumerate(all_masks):
        plt.plot(mask_traj, label=f"{str(i).zfill(2)}")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"{video_dir}/masks.png")



def main(_):
    wandb.init(project='real_franka_reach')
    wandb.config.update(FLAGS)

    from gym.wrappers.time_limit import TimeLimit
    from reward_extraction.reward_functions import RobotLearnedRewardFunction
    from robot.franka_env import SimpleRealFrankReach, LrfRealFrankaReach, LrfCabinetDoorOpenFranka
    from robot.utils import HZ

    # defaults
    use_gripper, use_camera, use_r3m, obs_key = False, False, False, None
    MAX_EPISODE_TIME_S = 15
    MAX_STEPS = HZ * MAX_EPISODE_TIME_S

    # exp_str = '013023_real_franka_reach'
    # exp_str = '013023_reach_with_gripper_and_camera'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_vec"
    # exp_str = 'r3m50_experiment'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_vec"
    # exp_str = '020123_couscous_reach'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_vec"
    # exp_str = '020123_couscous_reach_rlwithppc'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"
    # exp_str = '020123_couscous_reach_rlwithppc_bigsteps'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"
    # exp_str = '020223_couscous_reach_rlwithppc_bigsteps_and_rankinginit'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"
    # exp_str = '021723_debugnewcode'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"
    # exp_str = '022023_yogablock'; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"
    exp_str = "022623_cabinet_open"; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"
    # exp_str = "codetest"; use_gripper, use_camera, use_r3m, obs_key = False, True, True, "r3m_with_ppc"

    repo_root = Path.cwd()
    exp_dir = f'{repo_root}/walk_in_the_park/saved/{exp_str}'

    # load r3m here so multiple things can use it without having multiple resnets loaded into GPU memory
    r3m_net = load_r3m("resnet50")
    # r3m_net = load_r3m("resnet18")
    r3m_net.to("cuda")
    r3m_net.eval()
    r3m_embedding_dim = 2048 #512 #2048

    if FLAGS.real_robot:
        env = LrfCabinetDoorOpenFranka(
            home="default",
            hz=HZ,
            controller="cartesian",
            mode="default",
            use_camera=use_camera,
            use_gripper=use_gripper,
            use_r3m=use_r3m,
            r3m_net=r3m_net,
            only_pos_control=True,
            random_reset_home_pose=False,
        )
        # env = LrfRealFrankaReach(
        #     home="default",
        #     hz=HZ,
        #     controller="cartesian",
        #     mode="default",
        #     use_camera=use_camera,
        #     use_gripper=use_gripper,
        #     use_r3m=use_r3m,
        #     r3m_net=r3m_net,
        #     only_pos_control=True,
        #     random_reset_home_pose=True,
        # )
        # env = SimpleRealFrankReach(
        #     goal=np.array([0.68, 0.0, 0.4]),
        #     home="default",
        #     hz=HZ,
        #     controller="cartesian",
        #     mode="default",
        #     use_camera=use_camera,
        #     use_gripper=use_gripper,
        #     use_r3m=use_r3m,
        #     only_pos_control=True
        # )
    else:
        assert False # what are you doing

    env = wrap_gym(env, rescale_actions=True, obs_key=obs_key)

    env = TimeLimit(env, MAX_STEPS)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)


    chkpt_dir = f'{exp_dir}/checkpoints'
    # if os.path.exists(exp_dir):
    #     shutil.rmtree(exp_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = f'{exp_dir}/buffers'
    os.makedirs(buffer_dir, exist_ok=True)
    img_buffer_path = f'{buffer_dir}/image_buffer.hdf'

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps,
                                     image_shape=None,
                                     image_disk_save_path=img_buffer_path)
        img_replay_buffer = RAMImageReplayBuffer(capacity=50_000, img_shape=env.image_space.shape)
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

    '''
    setup learned reward function
    '''
    lrf = RobotLearnedRewardFunction(
        obs_size=r3m_embedding_dim,
        exp_dir=exp_dir,
        demo_path=f"{FLAGS.demo_dir}/demos.hdf",
        replay_buffer=replay_buffer,
        image_replay_buffer=img_replay_buffer,
        horizon=MAX_STEPS,
        r3m_net=r3m_net,
    )
    if last_checkpoint is not None:
        lrf.load_models()
    env.set_lrf(lrf)


    '''
    train video recorder to see how live data is being evaluated
    '''
    train_video_dir = Path(f"{exp_dir}/train")
    if not os.path.exists(str(train_video_dir)):
        os.makedirs(str(train_video_dir))
    train_recorder = VideoRecorder(save_dir=train_video_dir, fps=env.hz)
    progresses, masks, rewards = [], [], []

    observation, done = env.reset(), False
    image = env.rgb

    train_recorder.init(image)
    rewards = []

    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        next_image = env.rgb
        progress, mask, reward = env.lrf.last_pmr()
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
            plt.clf(); plt.cla()
            plt.plot(progresses, label="progress")
            plt.plot(masks, label='mask')
            plt.plot(rewards, label='reward')
            plt.ylim(0, 1)
            plt.legend()
            plt.savefig(f"{train_video_dir}/{save_str}_pmr.png")

            '''
            run some eval episodes if we haven't run one in a while
            '''
            if i - last_eval_idx > FLAGS.eval_interval and i >= FLAGS.start_training:
                eval(env, agent, exp_dir, i)
                last_eval_idx = i

            '''
            proper reset of the environment and some logging
            '''
            observation, done = env.reset(), False
            image = env.rgb
            progresses, masks, rewards = [], [], []
            train_recorder.init(image)

            for k, v in info['episode'].items():
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                wandb.log({f'training/{decode[k]}': v}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)

            # start = time.time()
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            # end = time.time()
            # print(f"{end - start} seconds to update agent")

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)

            if i % FLAGS.lrf_update_frequency == 0:
                lrf.train(FLAGS.utd_ratio)

            if i % (FLAGS.lrf_update_frequency * 2) == 0:
                lrf.eval_lrf()

        if i % FLAGS.checkpoint_interval == 0 and i > 0:
            checkpoints.save_checkpoint(chkpt_dir,
                                        agent,
                                        step=i + 1,
                                        keep=20,
                                        overwrite=True)

            if lrf._seen_on_policy_data:
                lrf.save_models()
                lrf.eval_lrf()

            try:
                shutil.rmtree(buffer_dir)
            except:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                pickle.dump(replay_buffer, f)
            img_replay_buffer.save_to_disk(buffer_dir)


if __name__ == '__main__':
    app.run(main)
