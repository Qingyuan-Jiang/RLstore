# Training with DQN on 'CartPole-v1' environment.
# Qingyuan Jiang. May. 26th. 2020
#

if __name__ == '__main__':
    import torch
    from DoubleDQN.train_doubledqn import doubledqn_algo as algo

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Training parameters.
    algo(env_name='CartPole-v1', device=device,
         steps_per_epoch=1000, epochs=100, max_ep_len=1000, replay_size=int(5e4), batch_size=256,
         start_steps=1000, update_after=1000, update_every=20,
         gamma=0.999, epsilon_start=0.8, epsilon_final=0.05, polyak=0.995, save_freq=5)
