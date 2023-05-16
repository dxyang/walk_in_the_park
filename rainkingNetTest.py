from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
from reward_extraction.models import R3MPolicy
from torch import optim
from robot.data import RoboDemoDset
class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_list):
        self.trajs = trajectory_list
        self.num_trajs = len(trajectory_list)
        self.horizon = len(trajectory_list[0])
        self.transforms = T.Compose([
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),  # divides by 255, will also convert to chw
            ])

    def __len__(self):
        return self.num_trajs * self.horizon

    def __getitem__(self, idx):
        traj_num = np.random.randint(self.num_trajs)

        ts = np.random.randint(self.horizon, size=(2,))
        t0 = ts[0]
        t1 = ts[1]

        if t0 > t1:
            label = np.array([1, 0])
        else:
            label = np.array([0, 1])

        state_t0 = self.transforms(self.trajs[traj_num][t0])
        state_t1 = self.transforms(self.trajs[traj_num][t1])

        return {
            "t0": state_t0, 
            "t1": state_t1, 
            "label": label,
        }
    
    def inOrder(self, traj_idx, idx):
        return self.transforms(self.trajs[traj_idx][idx])


demo_dir = '/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/obstacle-push'

expert_data = RoboDemoDset(f"{demo_dir}/demos.hdf", read_only_if_exists=True)
trajectories = expert_data[:]["rgb"]
batch_size = 64
training_data = TrajectoryDataset(trajectories)
device = torch.device("cuda")
train_dataloader = DataLoader(training_data, batch_size=batch_size)
net = R3MPolicy().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()
# import pdb
# pdb.set_trace()
num_epochs = 50
losses = []
for e_idx in tqdm(range(num_epochs)):
    for data_dict in train_dataloader:
        state_t0 = data_dict["t0"].to(device).float()
        state_t1 = data_dict["t1"].to(device).float()
        labels_0 = data_dict["label"][:, 0].to(device).float()
        labels_1 = data_dict["label"][:, 1].to(device).float()
        optimizer.zero_grad()
        

        # preprocess images
        # blah

        # do inference, clear optimizer, zerograd, whatever
        logits_t0 = net(state_t0).squeeze()
        logits_t1 = net(state_t1).squeeze()

        # calculate loss
        loss = criterion(logits_t0, labels_0)
        loss += criterion(logits_t1, labels_1)

        # backprop
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
plt.plot([i for i in range(len(losses))], losses)
plt.show()
plt.savefig(f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/saved/plotsbro/loss.png")
with torch.no_grad():
    net.eval()
    for demonum in range(len(expert_data)):
        xs = []
        ys = []
        for i in range(len(trajectories[demonum])):
            image = training_data.inOrder(demonum, i).to(device).float()
            pred = torch.sigmoid(net(image.unsqueeze(0))).item()
            xs.append(i)
            ys.append(pred)
        
        plt.cla(); plt.clf()
        plt.plot(xs, ys)
        plt.savefig(f'/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/walk_in_the_park/saved/plotsbro/demo{demonum}')

