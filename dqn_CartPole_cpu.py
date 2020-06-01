# Training with DQN on 'CartPole-v1' environment.
# Qingyuan Jiang. May. 26th. 2020
#

if __name__ == '__main__':
    import torch
    from dqn.train_dqn import dqn_algo

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Training parameters.
    # Changed gym/__init__.py the environment "MountainCarContinuous-v1', max_episode_steps=10000
    dqn_algo(env_name='CartPole-v1', device=device,
             steps_per_epoch=200, epochs=800, max_ep_len=500, replay_size=int(1e4), batch_size=128,
             start_steps=200, update_after=200, update_every=20, target_freq=10,
             gamma=0.99, epsilon_start=0.8, epsilon_final=0.05, save_freq=5)
