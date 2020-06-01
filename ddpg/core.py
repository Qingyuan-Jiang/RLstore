# RL algorithm exercise.
# Deep Deterministic Policy Gradient.
# Qingyuan Jiang. Mar. 2nd. 2020

import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt


def mlp(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


class MLPActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.P = mlp(pi_sizes, activation, torch.nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.P(obs)


class MLPQFunction(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.Q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.Q(torch.cat([obs, act], dim=-1).cuda())
        return torch.squeeze(q, -1)  # Critical to ensure Q has right shape.


class DDPGAgent:
    def __init__(self, obsDim, actionDim, q_hidden_sizes, q_activation, p_hidden_sizes, p_activation, device=None,
                 gamma=.9, p_lr=1e-3, q_lr=1e-3, polyak=0.995, act_noise=1e-1, act_limit=None,
                 num_test_episodes=10, max_test_ep_len=1000, logger_kwargs=dict()):

        # Save Neural Network information
        self.obsDim = obsDim
        self.actionDim = actionDim
        self.q_hidden_sizes = q_hidden_sizes
        self.q_activation = q_activation
        self.p_hidden_sizes = p_hidden_sizes
        self.p_activation = q_activation
        self.device = device
        self.act_limit_clip = torch.from_numpy(act_limit[1]).to(self.device)

        # Create Neural Networks
        self.P_targ = MLPActor(obsDim, actionDim, p_hidden_sizes, p_activation, self.act_limit_clip)
        self.P_curr = MLPActor(obsDim, actionDim, p_hidden_sizes, p_activation, self.act_limit_clip)
        self.Q_targ = MLPQFunction(obsDim, actionDim, q_hidden_sizes, q_activation)
        self.Q_curr = MLPQFunction(obsDim, actionDim, q_hidden_sizes, q_activation)

        if device is not None:
            self.P_targ.cuda()
            self.P_curr.cuda()
            self.Q_targ.cuda()
            self.Q_curr.cuda()

        # Save hyper-parameters for learning
        self.gamma = gamma
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.polyak = polyak
        self.act_noise = act_noise
        self.act_limit = act_limit

        # Save hyper-parameters for test
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_test_ep_len
        self.test_epoch = 0
        self.test_ret = []
        self.test_len = []
        self.test_ret_win = []
        self.test_win_len = 10
        # self.test_fig, self.test_ax = plt.subplots()
        plt.title('Agent test with deterministic policy.')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Returns')

        # Set up optimizers for policy and Q-function
        self.P_optimizer = Adam(self.P_curr.parameters(), lr=self.p_lr)
        self.Q_optimizer = Adam(self.Q_curr.parameters(), lr=self.q_lr)

        # 
        for p_targ in self.P_targ.parameters():
            p_targ.requires_grad = False

        for q_targ in self.Q_targ.parameters():
            q_targ.requires_grad = False

        # Logger
        # self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

    def action(self, obs, noise=True):
        obs = torch.from_numpy(obs).to(self.device)
        a = self.P_curr(obs).detach()
        a = a.cpu().numpy()
        if noise:
            a = a + self.act_noise * np.random.randn(len(a))
        if self.act_limit is not None:
            a_low, a_high = self.act_limit[0], self.act_limit[1]
            a = np.clip(a, a_low, a_high)
        return a

    def compute_loss_q(self, batch):
        # Computing DDPG Q-loss
        # Compute update target with actor and critic estimate.

        obs, a, r, obs_next, done = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']

        obs, a, obs_next = obs.to(self.device), a.to(self.device), obs_next.to(self.device)
        r, done = r.to(self.device), done.to(self.device)

        q = self.Q_curr(obs, a)
        with torch.no_grad():
            q_pi_targ = self.Q_targ(obs_next, self.P_targ(obs_next))
            y = r + self.gamma * (1 - done) * q_pi_targ

        # Mean-Squared Bellman Error (MSBE)
        loss_q = ((q - y) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())
        # print("Update Q function with loss: ", loss_q.detach())
        return loss_q, loss_info

    def compute_loss_p(self, batch):
        obs = batch['obs']
        obs = obs.to(self.device)
        loss_pi = - self.Q_curr(obs, self.P_curr(obs)).mean()
        # print("Update P function with loss: ", loss_pi.detach())
        return loss_pi

    def update(self, batch):
        # First run one gradient descent step for Q.
        self.Q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(batch)
        loss_q.backward()
        self.Q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for q in self.Q_curr.parameters():
            q.requires_grad = False

        # Next run one gradient descent step for P (policy).
        self.P_optimizer.zero_grad()
        loss_p = self.compute_loss_p(batch)
        loss_p.backward()
        self.P_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for q in self.Q_curr.parameters():
            q.requires_grad = True

        # Record things
        # self.logger.store(LossQ=loss_q.item(), LossPi=loss_p.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p_curr, p_targ in zip(self.P_curr.parameters(), self.P_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p_curr.data)

            for q_curr, q_targ in zip(self.Q_curr.parameters(), self.Q_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                q_targ.data.mul_(self.polyak)
                q_targ.data.add_((1 - self.polyak) * q_curr.data)

    def test_agent(self, test_env, render=False):
        test_ret, test_len = [], []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(self.action(o, noise=False))
                ep_ret += r
                ep_len += 1
                if render:
                    test_env.render()
            # self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            test_ret.append(ep_ret)
            test_len.append(ep_len)
        avg_ret = np.average(np.array(test_ret))
        avg_len = np.average(np.array(test_len))

        self.test_epoch = self.test_epoch + 1
        print("##### Epoch", self.test_epoch, "test rets: ", avg_ret, "with epi. length: ", avg_len, "#####")
        self.test_ret.append(avg_ret)
        self.test_len.append(avg_len)
        if len(self.test_ret) <= self.test_win_len:
            self.test_ret_win.append(np.average(self.test_ret))
        else:
            ret_win = self.test_ret[-self.test_win_len:]
            self.test_ret_win.append(np.average(ret_win))
        x = np.arange(len(self.test_ret))
        plt.scatter(x, self.test_ret, c='green', marker='o', label='Test returns')
        plt.plot(self.test_ret_win, '-b', label='Average returns', linewidth=2)
        plt.pause(0.05)

    '''
        def log_info(self, epoch, steps, time_delta):
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', steps)
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Time', time_delta)
        self.logger.dump_tabular()
    '''


if __name__ == '__main__':
    import gym

    # env_name = 'FetchPickAndPlace-v1'
    env_name = 'FetchPickAndPlace-v1'
    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound_high = env.action_space.high
    act_bound_low = env.action_space.low
    act_bound = np.array([act_bound_low, act_bound_high])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ag = DDPGAgent(obs_dim, act_dim,
                   q_hidden_sizes=(64, 64), q_activation=torch.nn.ReLU, p_hidden_sizes=(64, 64),
                   p_activation=torch.nn.ReLU, device=device,
                   polyak=0.995, p_lr=1e-3, q_lr=1e-3, act_noise=1e-1, act_limit=act_bound,
                   num_test_episodes=5, max_test_ep_len=20, logger_kwargs=dict())

    for _ in range(10):
        ag.test_agent(env, render=True)

    plt.show()
    env.close()
