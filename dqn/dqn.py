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
            self.layer1 = torch.nn.Linear(obs_dim + act_dim, 16)
            self.layer2 = torch.nn.Linear(16, 16)
            self.layer3 = torch.nn.Linear(16, 16)
            self.layer4 = torch.nn.Linear(16, 1)
            self.device = device

        def forward(self, obs, act):
            if not self.device == 'cpu':
                obs = obs.cuda()
                act = act.cuda()
            out1 = Function.relu(self.layer1(torch.cat([obs, act], dim=-1)))
            out2 = Function.relu(self.layer2(out1))
            out3 = Function.relu(self.layer3(out2))
            qvalue = self.layer4(out3)
            return qvalue

    def __init__(self, obs_dim, act_dim, act_mesh, device, gamma):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_mesh = act_mesh

        self.Q = self.MLPQFunction(obs_dim, act_dim)
        self.Q_targ = self.MLPQFunction(obs_dim, act_dim)

        self.Q_targ.load_state_dict(self.Q.state_dict())
        self.Q_targ.eval()

        for q in self.Q_targ.parameters():
            q.requires_grad = False

        self.Q_optimizer = Adam(self.Q.parameters(), lr=1e-4)

        self.gamma = gamma

        self.device = device
        if device != "cpu":
            self.Q.cuda()
            self.Q_targ.cuda()

        return

    def action(self, obs):
        obs = torch.from_numpy(obs).float()
        obs = obs.to(self.device)

        qBest = - np.inf
        aBest = self.act_mesh[0]
        with torch.no_grad():
            for a in self.act_mesh:
                q = self.Q(obs, a).detach()
                q = q.cpu().numpy()
                qBest = q if q >= qBest else qBest
                aBest = a if q >= qBest else aBest
        return aBest

    def update(self, batch):
        self.Q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss(batch)
        loss_q.backward()
        self.Q_optimizer.step()

    def compute_loss(self, batch):

        obs, a, r, obs_next, done = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']

        obs, a, obs_next = obs.to(self.device), a.to(self.device), obs_next.to(self.device)
        r, done = r.to(self.device), done.to(self.device)

        q = self.Q(obs, a)

        with torch.no_grad():
            batch_size, _ = obs.shape
            q_targ = - np.inf * torch.ones(batch_size).view(-1, 1)
            for act in self.act_mesh:
                act = act * torch.ones(batch_size).view(-1, 1)
                q_ = self.Q_targ(obs, act)
                q_targ[q_ > q_targ] = q_[q_ > q_targ]

            y = r + self.gamma * (1 - done) * q_targ.flatten()

        loss = Function.mse_loss(q, y)
        # loss = ((q - y) ** 2).mean()
        loss_info = dict(Qvals=q.detach().cpu().numpy())

        return loss, loss_info
