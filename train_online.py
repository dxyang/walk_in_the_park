#! /usr/bin/env python
import os
import pickle
import shutil

import numpy as np
import tqdm

import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('checkpoint_interval', 1000, 'Checkpoing interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of training steps to start training.')
# flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
# flags.DEFINE_integer('action_history', 1, 'Action history.')
# flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', True, 'Use real robot.')
config_flags.DEFINE_config_file(
    'config',
    'walk_in_the_park/configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    wandb.init(project='real_franka_reach')
    wandb.config.update(FLAGS)

    from robot.utils import HZ

    exp_str = '013023_real_franka_reach'

    if FLAGS.real_robot:
        from robot.env import SimpleRealFrankReach
        env = SimpleRealFrankReach(
            goal=np.array([0.8, 0.0, 0.4]),
            home="default",
            hz=HZ,
            controller="cartesian",
            mode="default",
            use_camera=False,
            use_gripper=False,
        )
    else:
        assert False # what are you doing

    env = wrap_gym(env, rescale_actions=True)

    from gym.wrappers.time_limit import TimeLimit
    MAX_EPISODE_TIME_S = 30
    MAX_STEPS = HZ * MAX_EPISODE_TIME_S
    env = TimeLimit(env, MAX_STEPS)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # env = gym.wrappers.RecordVideo(
    #     env,
    #     f'videos/train_{FLAGS.action_filter_high_cut}',
    #     episode_trigger=lambda x: True)
    env.seed(FLAGS.seed)


    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)

    exp_dir = f'walk_in_the_park/saved/{exp_str}'
    chkpt_dir = f'{exp_dir}/checkpoints'
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = f'{exp_dir}/buffers'

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        replay_buffer.seed(FLAGS.seed)
    else:
        start_i = int(last_checkpoint.split('_')[-1])

        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                wandb.log({f'training/{decode[k]}': v}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)

        if i % FLAGS.checkpoint_interval == 0:
            checkpoints.save_checkpoint(chkpt_dir,
                                        agent,
                                        step=i + 1,
                                        keep=20,
                                        overwrite=True)

            try:
                shutil.rmtree(buffer_dir)
            except:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                pickle.dump(replay_buffer, f)


if __name__ == '__main__':
    app.run(main)
