# Training with DDPG on 'MountainCarContinuous-v0' environment.
# Qingyuan Jiang. Mar. 10th. 2020
# 

if __name__ == '__main__':
    import torch
    from ddpg.train import train

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training parameters.
    # Changed gym/__init__.py the environment "MountainCarContinuous-v1', max_episode_steps=10000
    train(env_name='MountainCarContinuous-v0', render=False, device=device,
          steps_per_epoch=20000, epochs=100, max_ep_len=5000, replay_size=int(2e5), batch_size=200,
          start_steps=50000, update_after=4000, update_every=50,
          hidden_sizes=(128, 128, 128), activation=torch.nn.ReLU,
          gamma=0.9999, polyak=0.995, p_lr=1e-3, q_lr=1e-3, act_noise=0.1,
          num_test_episodes=5, max_test_ep_len=2000, logger_kwargs=dict(), save_freq=1)
