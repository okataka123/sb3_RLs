from stable_baselines3 import PPO, DQN, A2C, SAC

class config:
    # total_timesteps = 512000
    # total_timesteps = 128000
    total_timesteps = 64000

    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    env_name = 'SimpleBattileShip'

    algo_dict = {
        'PPO': PPO,
        'DQN': DQN,
        'A2C': A2C,
        'SAC': SAC,
    }