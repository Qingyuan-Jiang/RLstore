# RL algorithm exercise.
# Deep Deterministic Policy Gradient.
# Qingyuan Jiang. Mar. 2nd. 2020

import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as Function
import matplotlib.pyplot as plt


class dqnAgent:
    class MLPQFunction(torch.nn.Module):

        def __init__(self, obs_dim, act_dim, device='cpu'):
            super().__init__()
            self.layer1 = torch.nn.Linear(obs_dim, 64)
            self.layer2 = torch.nn.Linear(64, 64)
            self.layer3 = torch.nn.Linear(64, 64)
            self.layer4 = torch.nn.Linear(64, act_dim)
            self.device = device

        def forward(self, obs):
            if not self.device == 'cpu':
                obs = obs.cuda()
            out1 = Function.relu(self.layer1(obs))
            out2 = Function.relu(self.layer2(out1))
            out3 = Function.relu(self.layer3(out2))
            qvalue = self.layer4(out3)
            return qvalue

    def __init__(self, obs_dim, act_dim, device, lr, gamma,
                 num_test_episodes=10, max_ep_len=1000):

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.Q = self.MLPQFunction(obs_dim, act_dim, device=device)
        self.Q_targ = self.MLPQFunction(obs_dim, act_dim, device=device)

        self.Q_targ.load_state_dict(self.Q.state_dict())
        self.Q_targ.eval()

        for q in self.Q_targ.parameters():
            q.requires_grad = False

        self.Q_optimizer = Adam(self.Q.parameters(), lr=lr)

        self.gamma = gamma

        self.device = device
        if device != "cpu":
            self.Q.cuda()
            self.Q_targ.cuda()

        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len

    def action(self, obs):
        obs = torch.from_numpy(obs).float()
        obs = obs.to(self.device)
        # q_list = self.Q(obs).detach().cpu().numpy()
        # aBest = np.argmax(q_list)
        q_list = self.Q(obs).detach().cpu()
        act = q_list.max(0)[1]
        return act.numpy()

    def update(self, batch):
        self.Q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss(batch)
        loss_q.backward()
        self.Q_optimizer.step()

    def compute_loss(self, batch):
        obs, a, r, obs_next, done = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']
        obs, a, obs_next = obs.to(self.device), a.to(self.device), obs_next.to(self.device)
        r, done = r.to(self.device), done.to(self.device)

        q = self.Q(obs).gather(1, a.long().view(-1, 1)).flatten()

        with torch.no_grad():
            q_targ = self.Q_targ(obs).max(1)[0].detach()
            y = r + self.gamma * (1 - done) * q_targ.flatten()

        loss = Function.mse_loss(q, y)
        loss_info = dict(Qvals=q.detach().cpu().numpy())
        return loss, loss_info

    def test_agent(self, test_env, render=False):
        test_ret, test_len = [], []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                o, r, d, _ = test_env.step(self.action(o))
                ep_ret += r
                ep_len += 1
                if render:
                    test_env.render()
            test_ret.append(ep_ret)
            test_len.append(ep_len)
        avg_ret = np.average(np.array(test_ret))
        avg_len = np.average(np.array(test_len))
        return avg_ret, avg_len
