# Model evaluation
# Jun. 2nd. 2020, Qingyuan Jiang

import gym
from

if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    obs_dim = 2
    act_dim = env.action_space.n

    obs_bound_low = env.observation_space.low
    obs_bound_high = env.observation_space.high


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