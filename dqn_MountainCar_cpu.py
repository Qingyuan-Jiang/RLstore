# Training with DQN on 'CartPole-v1' environment.
# Qingyuan Jiang. May. 26th. 2020
#
from dqn.train_dqn import ReplayBuffer
from dqn.dqn import dqnAgent
import gym
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import math
from copy import deepcopy

if __name__ == '__main__':

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Params
    replay_size = int(1e4)
    batch_size = 256
    start_steps = 100
    update_after = 100
    update_every = 20
    target_freq = 5
    save_freq = 5

    gamma = 0.999
    epsilon_start = 0.8
    epsilon_final = 0.005

    steps_per_epoch = 100
    epochs = 20
    max_ep_len = 1000
    save_path = 'model/'

    env = gym.make('MountainCar-v0')
    env_test = deepcopy(env)

    obs_dim = 2
    act_dim = env.action_space.n

    ag = dqnAgent(obs_dim, act_dim, device, gamma)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1, size=replay_size)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    PATH = save_path + 'dqn_cuda.pt' if device is not 'cpu' else save_path + 'dqn_cpu.pt'
    obs, ep_ret, ep_len = env.reset(), 0, 0
    # x = encoder(obs)
    epsilon = epsilon_start

    '''
    if path.exists(PATH):
        print("Loading model from path: " + PATH)
        ag.Q.load_state_dict(torch.load(PATH))
        ag.Q.eval()
        ag.Q_targ.load_state_dict(ag.Q.state_dict())
        ag.Q_targ.eval()
    '''

    # EP_ret, traj_ret = [], []
    test_ret, test_len = [], []
    test_win_len = 10
    test_ret_win = []

    # fig, ax = plt.subplots()
    # ax.set_title('Agent Returns of the epoch.')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Avg Returns')
    # ax.grid(True)

    for t in range(total_steps):

        # Observe state s and select an action.
        if t > start_steps:
            if np.random.uniform(0, 1) < epsilon:
                a = env.action_space.sample()
            else:
                a = ag.action(obs)
                # a = int(ag.action(x))
        else:
            a = env.action_space.sample()

        # Execute a in the environment.
        obs_next, r, d, info = env.step(a)
        # x_next = encoder(obs_next)

        ep_ret = ep_ret + r
        # ep_len = ep_len + 1
        env.render()

        # d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(obs, a, r, obs_next, d)
        # replay_buffer.store(x, a, r, x_next, d)

        obs = obs_next
        # x = x_next

        # End of trajectory
        if d or (ep_len == max_ep_len):
            # obs, ep_ret, ep_len = env.reset(), 0, 0
            obs = env.reset()
            # x = encoder(obs)
            # traj_ret.append(ep_ret)
            # ep_ret = 0

        # Update of the agent
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                ag.update(batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                torch.save(ag.Q.state_dict(), PATH)

            # if epoch % target_freq == 0:
            #     ag.Q_targ.load_state_dict(ag.Q.state_dict())
            #     # ag.Q_targ.eval()

            # EP_ret.append(np.average(traj_ret))
            # ax.plot(EP_ret, 'b')
            # plt.pause(0.001)

            # traj_ret = []

            epsilon = epsilon_final + math.exp(-1 * epoch / epochs) * (epsilon_start - epsilon_final)

            # self.test_epoch = self.test_epoch + 1
            # print("##### Epoch", self.test_epoch, "test rets: ", avg_ret, "with epi. length: ", avg_len, "#####")

            avg_ret, avg_len = ag.test_agent(env_test)
            test_ret.append(avg_ret)
            test_len.append(avg_len)

            if len(test_ret) <= test_win_len:
                test_ret_win.append(np.average(test_ret))
            else:
                ret_win = test_ret[-test_win_len:]
                test_ret_win.append(np.average(ret_win))
            x = np.arange(len(test_ret))
            plt.scatter(x, avg_ret, c='green', marker='o', label='Test returns')
            plt.plot(test_ret_win, '-b', label='Average returns', linewidth=2)
            plt.pause(0.001)
            plt.axis(True)

            # Log info about epoch
            print("##### Epoch: %i at step t: %i after time %i mins" % (epoch, t + 1, (time.time() - start_time) / 60))
    plt.show()
    env.close()
