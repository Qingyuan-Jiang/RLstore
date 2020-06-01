# Training with DDPG on FetchReach-v1 environment.


if __name__ == '__main__':
    from ddpg.train import train
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Changed gym/__init__.py the environment "FetchReach-v1", max_episode_steps=5000
    train(env_name='FetchReach-v1', render=False, device=device,
          steps_per_epoch=2000, epochs=1000, max_ep_len=100, replay_size=int(2e5), batch_size=200,
          start_steps=40000, update_after=4000, update_every=50,
          hidden_sizes=(256, 256, 256), activation=torch.nn.ReLU,
          gamma=0.995, polyak=0.995, p_lr=1e-3, q_lr=1e-3, act_noise=0.1,
          num_test_episodes=10, max_test_ep_len=500, logger_kwargs=dict(), save_freq=1)
