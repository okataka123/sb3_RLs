from stable_baselines3 import PPO, DQN, A2C, SAC

class Config:
    # total_timesteps = 1024000
    # total_timesteps = 512000
    # total_timesteps = 256000
    # total_timesteps = 128000
    total_timesteps = 64000

    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    env_name = 'SimpleBattleShip'

    algo_dict = {
        'PPO': PPO,
        'DQN': DQN,
        'A2C': A2C,
        'SAC': SAC,
    }

class Field_Config:
    # フィールドサイズ
    grid_size = 8
    # 船の種類数
    num_ship_kind = 1
    # 陣形パターン
    pattern = 1
    # 陣形パターンを平行移動するかどうか？
    random_translation = True