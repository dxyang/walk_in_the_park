import pickle
from rl.data import ReplayBuffer
import os
from reward_extraction.models import Policy, resnet18
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from pathlib import Path
from tqdm import tqdm
import os
from robot.data import RoboDemoDset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_root = Path.cwd()
exp_dir = f'{repo_root}/walk_in_the_park/saved/goalcat'

buffer_dir = "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/saved/new_push_sub/buffers/buffer_2016"

with open(buffer_dir, 'rb') as f:
            replay_buffer = pickle.load(f)

batch_size = 32

batch = replay_buffer.sample(batch_size=batch_size)
expert_data = RoboDemoDset("/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/push_sub/demos.hdf", read_only_if_exists=True)


cf_states_np = batch["observations"][:, :512]

obs_size = 1024
hidden_depth = 3
hidden_layer_size = 4096
same_traj_classifier = Policy(obs_size, 1, hidden_layer_size, hidden_depth, do_regularization=False) # # default is d_reg = False
same_traj_classifier.to(device)
same_traj_classifier.load_state_dict(torch.load(f"{exp_dir}/traj_classifier.pt"))
same_traj_classifier.eval()

true_classifier = Policy(512, 1, hidden_layer_size, hidden_depth, do_regularization=False) # # default is d_reg = False
true_classifier.to(device)
true_classifier.load_state_dict(torch.load(f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/saved/new_push_sub/checkpoints/002351/same_classifier_policy.pt"))
true_classifier.eval()
classifs = []
with torch.no_grad():
    traj_idx = 0
    for i, sample in enumerate(cf_states_np):
        if i == 0:
            continue
        sample = sample.reshape(1, 512)
        cur_state = np.append(sample,
                expert_data[traj_idx]["r3m_vec"][31][None], axis=1)
        cur_state = torch.Tensor(cur_state).float().to(device)
        classif = torch.sigmoid(same_traj_classifier(cur_state))
        classifs.append(classif.cpu().numpy()[0])

    plt.clf(); plt.cla()
    plt.plot([t for t in range(1, 32)], classifs)
    plt.savefig(f"{exp_dir}/classifying_traj_push_new.png")

classifs = []

with torch.no_grad():
    traj_idx = 0
    for i, sample in enumerate(cf_states_np):
        if i == 0:
            continue
        sample = sample.reshape(1, 512)
        cur_state = sample
        cur_state = torch.Tensor(cur_state).float().to(device)
        classif = torch.sigmoid(true_classifier(cur_state))
        classifs.append(classif.cpu().numpy()[0])

    plt.clf(); plt.cla()
    plt.plot([t for t in range(1, 32)], classifs)
    plt.savefig(f"{exp_dir}/classifying_traj_push_true.png")