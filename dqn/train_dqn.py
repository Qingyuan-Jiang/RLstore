# RL algorithm exercise.
# Deep Q-value Network
# Qingyuan Jiang. Mar. 27th. 2020

import numpy as np
import gym
import torch
from dqn.dqn import dqnAgent
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from os import path
import math


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        indx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[indx],
                     obs2=self.obs2_buf[indx],
                     act=self.act_buf[indx],
                     rew=self.rew_buf[indx],
                     done=self.done_buf[indx])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def encoder(obs):
    cart_position, cart_vel, pole_ang, pole_vel = obs
    pole_x_vel = np.sign(pole_ang) * np.cos(pole_ang) * pole_vel
    pole_y_vel = - np.sin(pole_ang) * pole_vel
    return np.array([cart_position, cart_vel, pole_ang, pole_vel, pole_x_vel, pole_y_vel])


def dqn_algo(env_name='CartPole-v1', device='cpu',
             steps_per_epoch=200, epochs=400, max_ep_len=500, replay_size=int(1e6), batch_size=100,
             start_steps=10000, update_after=200, update_every=50, target_freq=5,
             gamma=0.99, epsilon_start=0.9, epsilon_final=0.1, save_freq=5, save_path='model/'):
    env = gym.make(env_name)

    obs_dim = 6
    act_dim = 2

    ag = dqnAgent(obs_dim, act_dim, device, gamma)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1, size=replay_size)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    PATH = save_path + 'dqn_cuda.pt' if device is not 'cpu' else save_path + 'dqn_cpu.pt'
    obs, ep_ret, ep_len = env.reset(), 0, 0
    x = encoder(obs)
    epsilon = epsilon_start

    '''
    if path.exists(PATH):
        print("Loading model from path: " + PATH)
        ag.Q.load_state_dict(torch.load(PATH))
        ag.Q.eval()
        ag.Q_targ.load_state_dict(ag.Q.state_dict())
        ag.Q_targ.eval()
    '''

    EP_ret, traj_ret = [], []

    fig, ax = plt.subplots()
    ax.set_title('Agent Returns of the epoch.')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg Returns')
    ax.grid(True)

    for t in range(total_steps):

        # Observe state s and select an action.
        if t > start_steps:
            if np.random.uniform(0, 1) < epsilon:
                a = env.action_space.sample()
            else:
                # a = int(ag.action(obs))
                a = int(ag.action(x))
        else:
            a = env.action_space.sample()

        # Execute a in the environment.
        obs_next, r, d, info = env.step(a)
        x_next = encoder(obs_next)

        ep_ret = ep_ret + r
        # ep_len = ep_len + 1
        env.render()

        # d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        # replay_buffer.store(obs, a, r, obs_next, d)
        replay_buffer.store(x, a, r, x_next, d)

        # obs = obs_next
        x = x_next

        # End of trajectory
        if d or (ep_len == max_ep_len):
            # obs, ep_ret, ep_len = env.reset(), 0, 0
            obs = env.reset()
            x = encoder(obs)
            traj_ret.append(ep_ret)
            ep_ret = 0

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

            if epoch % target_freq == 0:
                ag.Q_targ.load_state_dict(ag.Q.state_dict())
                # ag.Q_targ.eval()

            EP_ret.append(np.average(traj_ret))
            ax.plot(EP_ret, 'b')
            plt.pause(0.001)

            traj_ret = []

            epsilon = epsilon_final + math.exp(-1 * epoch / epochs) * (epsilon_start - epsilon_final)
            # Log info about epoch
            print("##### Epoch: %i at step t: %i after time %i mins" % (epoch, t + 1, (time.time() - start_time) / 60))
    plt.show()
    env.close()
