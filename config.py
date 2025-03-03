from stable_baselines3 import PPO, DQN, A2C, SAC

class Config:
    # total_timesteps = 1024000
    total_timesteps = 512000
    # total_timesteps = 256000
    # total_timesteps = 128000
    # total_timesteps = 64000

    # env_name = 'CartPole-v1'
    # env_name = 'LunarLander-v2'
    # env_name = 'Acrobot-v1'
    # env_name = 'MountainCar-v0'
    env_name = 'SimpleBattleShip'

    algo_dict = {
        'DQN': DQN,
        'PPO': PPO,
        'A2C': A2C,
        'SAC': SAC,
    }

    # DQN params
    batch_size = 128
    seed = 0
    tau = 0.001

    learn_kwargs = {
        'DQN': {
            'policy': 'MlpPolicy',
            'batch_size': batch_size,
            'seed': seed,
            'tau': tau,
            'policy_kwargs': {
                'net_arch': [128, 128],
            },
        },
        'PPO': {},
        'A2C': {},
        'SAC': {},
    }


class Field_Config:
    # フィールドサイズ
    grid_size = 8
    # 船の種類数
    num_ship_kind = 1
    # 陣形パターン
    pattern = 4
    # 陣形パターンを平行移動するかどうか？
    random_translation = True


class Inference_Config:
    n_episode = 1
    save_Q_fig = True
    
    # pretrained_model_name = 'model_SimpleBattleShip_DQN_pat1_seed_0_random_False_64000_20250227-232357'
    pretrained_model_name = 'model_SimpleBattleShip_DQN_pat1_seed_0_random_True_512000_20250228-001659'
    # pretrained_model_name = 'model_Acrobot-v1_DQN_64000_20250301-221607'
    # pretrained_model_name = 'model_Acrobot-v1_DQN_256000_20250301-222022'
    # pretrained_model_name = 'model_MountainCar-v0_DQN_256000_20250301-235147'


if __name__ == '__main__':
    print(Config.learn_kwargs['DQN'])
    print(**Config.learn_kwargs['DQN'])
    