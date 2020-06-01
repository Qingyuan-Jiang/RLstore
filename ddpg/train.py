# RL algorithm exercise.
# Deep Deterministic Policy Gradient.
# Qingyuan Jiang. Mar. 2nd. 2020

import numpy as np
import gym
import torch
from ddpg import core
import matplotlib.pyplot as plt
import time
from copy import deepcopy


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


def train(env_name='FetchPickAndPlace-v1', render=False, device='cpu',
          steps_per_epoch=4000, epochs=100, max_ep_len=1000, replay_size=int(1e6), batch_size=100,
          start_steps=10000, update_after=1000, update_every=50,
          hidden_sizes=(64, 64, 64), activation=torch.nn.ReLU,
          gamma=0.99, polyak=0.995, p_lr=1e-3, q_lr=1e-3, act_noise=0.1,
          num_test_episodes=10, max_test_ep_len=1000, logger_kwargs=dict(), save_freq=5, save_path='/save'):
    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env_test = deepcopy(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound_high = env.action_space.high
    act_bound_low = env.action_space.low
    act_bound = np.array([act_bound_low, act_bound_high])

    ag = core.DDPGAgent(obs_dim, act_dim,
                        q_hidden_sizes=hidden_sizes, q_activation=activation, p_hidden_sizes=hidden_sizes,
                        p_activation=activation, device=device,
                        gamma=gamma, polyak=polyak, p_lr=p_lr, q_lr=q_lr, act_noise=act_noise, act_limit=act_bound,
                        num_test_episodes=num_test_episodes, max_test_ep_len=max_test_ep_len,
                        logger_kwargs=logger_kwargs)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ag.P, ag.Q])
    # ag.logger.log('\nNumber of parameters: \t P: %d, \t Q: %d\n' % var_counts)

    # Set up model saving
    # ag.logger.setup_pytorch_saver(ag.P_curr)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    PATH = save_path
    obs, ep_ret, ep_len = env.reset(), 0, 0

    for t in range(total_steps):

        # Observe state s and select action.
        if t > start_steps:
            a = ag.action(obs, noise=True)
        else:
            a = env.action_space.sample()

        # Execute a in the environment.
        obs_next, r, d, info = env.step(a)
        ep_ret = ep_ret + r
        ep_len = ep_len + 1
        if render:
            env.render()

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(obs, a, r, obs_next, d)

        obs = obs_next

        # End of trajectory
        if d or (ep_len == max_ep_len):
            # ag.logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, ep_ret, ep_len = env.reset(), 0, 0
            if d:
                print("Reset the env at step t: ", t, "at time ", int(time.time() - start_time), "because of d:", d)
            else:
                print("Reset the env at step t: ", t, "at time ", int(time.time() - start_time), "reaching the maximum "
                                                                                                 "epi. len: ",
                      max_ep_len)

        # Update of the agent
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                ag.update(batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch  # //: floor division. e.g. 9//2 = 4

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                torch.save(ag.Q_curr.state_dict(), PATH)
                torch.save(ag.P_curr.state_dict(), PATH)
            #     ag.logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            ag.test_agent(env_test)

            # Log info about epoch
            # ag.log_info(epoch, t, time.time() - start_time)
            print("##### Epoch", epoch, "at step t: ", t, "after time", int(time.time() - start_time))
    plt.show()
    env.close()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training parameters.
    # Only used for test. See 'train_mountcar' for training params.
    train(env_name='MountainCarContinuous-v0', render=False, device=device,
          steps_per_epoch=4000, epochs=100, max_ep_len=1000, replay_size=int(2e5), batch_size=200,
          start_steps=4000, update_after=1000, update_every=50,
          hidden_sizes=(128, 128, 128), activation=torch.nn.ReLU,
          gamma=0.9999, polyak=0.995, p_lr=1e-3, q_lr=1e-3, act_noise=0.1,
          num_test_episodes=10, max_test_ep_len=1000, logger_kwargs=dict(), save_freq=1)
