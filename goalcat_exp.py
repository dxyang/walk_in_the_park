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
os.makedirs(exp_dir, exist_ok=True)
demo_paths = ["/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/push_sub/demos.hdf",
              "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/obst_push_sub/demos.hdf",
              "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/open_sub/demos.hdf",
              "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/reach_kl/demos.hdf",
              "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/sweep_sub/demos.hdf",
              "/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/paper_draw_2/demos.hdf"]

class RankingNet():
    def __init__(self):
        obs_size = 1024
        hidden_depth = 3
        hidden_layer_size = 4096
        self.ranking_network = Policy(obs_size, 1, hidden_layer_size, hidden_depth)
        # self.ranking_network = resnet18()
        self.ranking_network = self.ranking_network.to(device)
        self.ranking_optimizer = optim.Adam(list(self.ranking_network.parameters()), lr=1e-4) 
        self.bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()
        self.batch_size = 32

        self.horizons = [32, 48, 32, 32, 48, 8]
        self.expert_data_paths = demo_paths
        self.expert_datas, self.num_expert_trajs_list = self.aggreg_expert_data()

    def aggreg_expert_data(self):
        expert_datas = []
        num_expert_trajs_list = []
        for expert_data_path in self.expert_data_paths:
            assert os.path.exists(expert_data_path)
            expert_data = RoboDemoDset(expert_data_path, read_only_if_exists=True)
            expert_datas.append(expert_data)
            num_expert_trajs_list.append(len(expert_data) - 1)

        return np.array(expert_datas), np.array(num_expert_trajs_list)

    def train_net(self, steps: int = 5000):
        '''
        train the ranking function a bit so it isn't oututting purely 0 during robot exploration
        '''
        ranking_init_losses = []
        num_steps = steps
        for i in tqdm(range(num_steps)):
            self.ranking_optimizer.zero_grad()

            ranking_loss, _ = self._train_ranking_step(plot_images=(i % 50 == 0))

            ranking_loss.backward()
            self.ranking_optimizer.step()
            ranking_init_losses.append(ranking_loss.item())

        torch.save(self.ranking_network.state_dict(), f"{exp_dir}/ranking_net.pt")

        # import IPython;
        # IPython.embed()

        # plot and save
        plt.clf(); plt.cla()
        plt.plot([t for t in range(num_steps)], ranking_init_losses)
        plt.savefig(f"{exp_dir}/ranking_init_loss.png")

    def process_data(self, demo_idxs, traj_idxs, t_idxs):
        res = []
        for (demo_idx, traj_idx, t_idx) in zip(demo_idxs, traj_idxs, t_idxs):
            goal_concated = np.append(self.expert_datas[demo_idx][traj_idx]["r3m_vec"][t_idx][None],
                        self.expert_datas[demo_idx][traj_idx]["r3m_vec"][self.horizons[demo_idx] - 1][None], axis=1)
            res.append(goal_concated)
        return np.concatenate(res)


    def _train_ranking_step(self, plot_images: bool = False):
        '''
        sample from expert data (factuals) => train ranking, classifier positives
        '''
        demo_idxs = np.random.randint(len(self.expert_datas), size=(self.batch_size,))
        expert_idxs = []
        expert_t_idxs = []
        expert_other_t_idxs = []
        for demo_idx in demo_idxs:
            expert_idxs.append(np.random.randint(self.num_expert_trajs_list[demo_idx]))
            expert_t_idxs.append(np.random.randint(low=1, high=self.horizons[demo_idx]))
            expert_other_t_idxs.append(np.random.randint(low=1, high=self.horizons[demo_idx]))
        expert_idxs = np.array(expert_idxs)
        expert_t_idxs = np.array(expert_t_idxs)
        expert_other_t_idxs = np.array(expert_other_t_idxs)
            
        labels = np.zeros((self.batch_size,))
        first_before = np.where(expert_t_idxs < expert_other_t_idxs)[0]
        labels[first_before] = 1.0 # idx is 1.0 if other timestep > timestep

        expert_states_t_np = self.process_data(demo_idxs, expert_idxs, expert_t_idxs)
        expert_states_other_t_np = self.process_data(demo_idxs, expert_idxs, expert_other_t_idxs)

        # import IPython;
        # IPython.embed()

        expert_states_t = torch.Tensor(expert_states_t_np).float().to(device)
        expert_states_other_t = torch.Tensor(expert_states_other_t_np).float().to(device)

        ranking_labels = F.one_hot(torch.Tensor(labels).long().to(device), 2).float()

        loss_monotonic = torch.Tensor([0.0])

        expert_logits_t = self.ranking_network(expert_states_t)
        expert_logits_other_t = self.ranking_network(expert_states_other_t)
        expert_logits = torch.cat([expert_logits_t, expert_logits_other_t], dim=-1)
        import IPython; IPython.embed()

        loss_monotonic = self.bce_with_logits_criterion(expert_logits, ranking_labels)

        return loss_monotonic, expert_states_t

    def eval(self):
        self.ranking_network.eval()
        progresses = []
        with torch.no_grad():
            for demo_idx, expert_data in enumerate(self.expert_datas):
                traj_idx = self.num_expert_trajs_list[demo_idx] - 1
                for t_idx in range(1, self.horizons[demo_idx]):
                    cur_state = np.append(expert_data[traj_idx]["r3m_vec"][t_idx][None],
                            expert_data[traj_idx]["r3m_vec"][self.horizons[demo_idx] - 1][None], axis=1)
                    cur_state = torch.Tensor(cur_state).float().to(device)
                    progress = torch.sigmoid(self.ranking_network(cur_state))
                    progresses.append(progress.cpu().numpy()[0])

                plt.clf(); plt.cla()
                plt.plot([t for t in range(1, self.horizons[demo_idx])], progresses)
                plt.savefig(f"{exp_dir}/ranking_progress_eval_{demo_idx}.png")
                progresses = []
        
        self.ranking_network.train()

def main():
    ranking_net = RankingNet()
    ranking_net.train_net(steps=5000)
    ranking_net.eval()

if __name__ == "__main__":
    main()

















































































