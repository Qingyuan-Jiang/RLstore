# Training with DQN on 'CartPole-v1' environment.
# Qingyuan Jiang. May. 26th. 2020
#

if __name__ == '__main__':
    import torch
    from dqn.train_dqn import dqn_algo

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Training parameters.
    dqn_algo(env_name='CartPole-v1', device=device,
             steps_per_epoch=1000, epochs=500, max_ep_len=1000, replay_size=int(1e5), batch_size=512,
             start_steps=5000, update_after=5000, update_every=20, target_freq=5,
             lr=1e-3, gamma=0.999, epsilon_start=0.8, epsilon_final=0.1, save_freq=5)
