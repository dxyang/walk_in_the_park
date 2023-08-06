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
        self.same_traj_classifier = Policy(obs_size, 1, hidden_layer_size, hidden_depth, do_regularization=False) # # default is d_reg = False
        self.same_traj_classifier.to(device)
        self.same_traj_optimizer = optim.Adam(list(self.same_traj_classifier.parameters()), lr=1e-4) # default is no weight decay
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
        classif_init_losses = []
        num_steps = steps
        for i in tqdm(range(num_steps)):
            self.same_traj_classifier.zero_grad()

            classif_loss = self._train_classif_step(plot_images=(i % 50 == 0))

            classif_loss.backward()
            self.same_traj_optimizer.step()
            classif_init_losses.append(classif_loss.item())

        torch.save(self.same_traj_classifier.state_dict(), f"{exp_dir}/traj_classifier.pt")

        # import IPython;
        # IPython.embed()

        # plot and save
        plt.clf(); plt.cla()
        plt.plot([t for t in range(num_steps)], classif_init_losses)
        plt.savefig(f"{exp_dir}/classif_init_loss.png")

    def process_data(self, demo_idxs, traj_idxs, t_idxs, counter: bool = False):
        if counter:
            demo_other_idxs = np.random.randint(len(self.expert_datas), size=(self.batch_size,))
        else:
            demo_other_idxs = demo_idxs
        res = []
        for (demo_idx, demo_other_idx, traj_idx, t_idx) in zip(demo_idxs, demo_other_idxs, traj_idxs, t_idxs):
            if counter:
                while demo_idx == demo_other_idx:
                    demo_other_idx = np.random.randint(len(self.expert_datas))
                traj_other_idx = np.random.randint(self.num_expert_trajs_list[demo_other_idx])
            else:
                traj_other_idx = traj_idx
            goal_concated = np.append(self.expert_datas[demo_idx][traj_idx]["r3m_vec"][t_idx][None],
                        self.expert_datas[demo_other_idx][traj_other_idx]["r3m_vec"][self.horizons[demo_other_idx] - 1][None], axis=1)
            res.append(goal_concated)
        return np.concatenate(res)


    def _train_classif_step(self, plot_images: bool = False):
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


        states_np = self.process_data(demo_idxs, expert_idxs, expert_t_idxs)
        states_counter_np = self.process_data(demo_idxs, expert_idxs, expert_other_t_idxs, counter=True)

        # import IPython;
        # IPython.embed()

        tr_states = torch.Tensor(states_np).float().to(device)
        cf_states = torch.Tensor(states_counter_np).float().to(device)

        classify_states = torch.cat([tr_states, cf_states], dim=0)
        traj_labels = torch.cat([torch.ones((tr_states.size()[0], 1)), torch.zeros((cf_states.size()[0], 1))], dim=0).to(device)

        traj_prediction_logits = self.same_traj_classifier(classify_states)
        loss_same_traj = self.bce_with_logits_criterion(traj_prediction_logits, traj_labels)

        return loss_same_traj

    def eval(self):
        self.same_traj_classifier.eval()
        classifs = []
        with torch.no_grad():
            for demo_idx, expert_data in enumerate(self.expert_datas):
                traj_idx = self.num_expert_trajs_list[demo_idx] - 1
                for t_idx in range(1, self.horizons[demo_idx]):
                    cur_state = np.append(expert_data[traj_idx]["r3m_vec"][t_idx][None],
                            expert_data[traj_idx]["r3m_vec"][self.horizons[demo_idx] - 1][None], axis=1)
                    cur_state = torch.Tensor(cur_state).float().to(device)
                    classif = torch.sigmoid(self.same_traj_classifier(cur_state))
                    classifs.append(classif.cpu().numpy()[0])

                plt.clf(); plt.cla()
                plt.plot([t for t in range(1, self.horizons[demo_idx])], classifs)
                plt.savefig(f"{exp_dir}/classifying_traj_eval_{demo_idx}.png")
                classifs = []
        
        self.same_traj_classifier.train()

def main():
    ranking_net = RankingNet()
    ranking_net.train_net(steps=1000)
    ranking_net.eval()

if __name__ == "__main__":
    main()

















































































