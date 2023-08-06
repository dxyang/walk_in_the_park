from pathlib import Path
from tqdm import tqdm
import cv2 as cv
import torch
import torchvision.transforms as T
from r3m import load_r3m
from PIL import Image
import numpy as np
import os
import subprocess
import time
import matplotlib.pyplot as plt
from reward_extraction.models import Policy


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



r3m = load_r3m("resnet18") # resnet18, resnet34
r3m.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),  # divides by 255, will also convert to chw
            ])

def eval_traj():
    data = np.load("/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/r2r_res.npy")
    goal=np.load("/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/R2R_data_end.npy"),
    clip = data
    end_frame = goal[0].reshape(1, 512)
    ranks = []
    masks = []
    progs = []

    for frame in clip:
        frame = frame.reshape(1, 512)
        frame = np.append(frame, end_frame, axis=1)
        frame = torch.Tensor(frame).float().to(device)
        with torch.no_grad():
            mask = torch.sigmoid(classify_net(frame))
            prog = torch.sigmoid(ranking_net(frame))
        rank = mask * prog
        ranks.append(rank.cpu().numpy()[0])
        masks.append(mask.cpu().numpy()[0])
        progs.append(prog.cpu().numpy()[0])

    results = dict()
    results['masks'] = masks
    results['progresses'] = progs
    results['rewards'] = ranks

    return results



def to_r3m(file):
    cap = cv.VideoCapture(str(file))
    fps = cap.get(cv.CAP_PROP_FPS)
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
        frames = r3m(frames * 255.0)
    frames = frames.squeeze().cpu().numpy()

    np.save(f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/r2r_res.npy", frames)

def calculate_traj_ranking(file_path):
    to_r3m(file_path)
    results = eval_traj()
    return results

