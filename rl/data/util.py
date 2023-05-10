'''
Add demo to the replay buffer
'''

import gym
from rlpd.data import ReplayBuffer
import os
from robot.data import RoboDemoDset
import time

def add_to_ds(env, replay_buffer, demo_dir):
    assert os.path.exists(demo_dir)
    expert_data = RoboDemoDset(demo_dir, read_only_if_exists=True)

    for traj in expert_data:
        eef_poses = traj['eef_pose']
        deltas = traj['ja']

        for cur in range(len(eef_poses) - 2):
            # TODO: This only works without gripper
            obs = eef_poses[cur]
            nxt_obs = eef_poses[cur+1]
            action = deltas[cur]
            rwd = env.calculate_reward(obs, exp_only=True)
            done = False 
            mask = 1.0
            replay_buffer.insert(
                dict(
                    observations=obs,
                    actions=action,
                    rewards=rwd,
                    masks=mask,
                    dones=done,
                    next_observations=nxt_obs,
                )
            )
        obs = nxt_obs
        action = deltas[cur+1]
        rwd = env.calculate_reward(obs, exp_only=True)
        done = True 
        mask = 0.0
        replay_buffer.insert(
            dict(
                observations=obs,
                actions=action,
                rewards=rwd,
                masks=mask,
                dones=done,
                next_observations=nxt_obs,
            )
        )

def update_RPB_reward(env, replay_buffer: ReplayBuffer, i):
    old_ds_dict = replay_buffer.dataset_dict
    time.sleep(0.1)

    #TODO batchify
    for i, obs in enumerate(old_ds_dict['next_observations'][:i]):
        old_ds_dict['rewards'][i] = env.calculate_reward(obs)

def plot_demo_gail(env, demo_dir, save_dir):
    assert os.path.exists(demo_dir)
    expert_data = RoboDemoDset(demo_dir, read_only_if_exists=True)

    for i, traj in enumerate(expert_data):
        eef_poses = traj['eef_pose']

        progresses = []
        masks = []
        rewards = []

        for cur in range(1, len(eef_poses) - 2):
            obs = eef_poses[cur]
            progress, mask, reward = env.debug_reward(obs)
            progresses.append(progress)
            masks.append(mask)
            rewards.append(reward)

        import matplotlib.pyplot as plt

        save_str = str(i).zfill(2)
        plt.clf(); plt.cla()
        plt.plot(progresses, label="progress")
        plt.plot(masks, label='mask')
        plt.plot(rewards, label='reward')
        plt.legend()
        plt.savefig(f"{save_dir}/{save_str}_pmr.png")
            
