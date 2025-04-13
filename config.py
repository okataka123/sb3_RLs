from stable_baselines3 import PPO, DQN, A2C, SAC

class Config:
    total_timesteps = 1024000
    # total_timesteps = 512000
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

    # T.B.D.
    params = {
        'DQN_MlpPolicy': {},
        'DQN_CNNPolicy': {},
    }

    # DQN params
    batch_size = 128
    n_node_layer1 = 128
    n_node_layer2 = 128
    n_node_layer3 = 128
    seed = 0
    tau = 0.001

    learn_kwargs = {
        'DQN': {
            'policy': 'MlpPolicy',
            'batch_size': batch_size,
            'seed': seed,
            'tau': tau,
            'policy_kwargs': {
                'net_arch': [n_node_layer1, n_node_layer2, n_node_layer3],
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
    pattern = 1
    # 陣形パターンを平行移動するかどうか？
    random_translation = True


class Inference_Config:
    n_episode = 1
    save_Q_fig = True
    
    # pretrained_model_name = 'model_SimpleBattleShip_DQN_pat1_seed_0_random_False_64000_20250227-232357'
    pretrained_model_name = 'model_SimpleBattleShip_DQN_pat1_seed_0_random_True_512000_20250228-001659'
    # pretrained_model_name = 'model_SimpleBattleShip_DQN_pat4_seed_0_random_True_512000_20250303-231407'
    # pretrained_model_name = 'model_SimpleBattleShip_DQN_pat5_seed_0_random_False_512000_20250308-183840' # 8×8
    # pretrained_model_name = 'model_SimpleBattleShip_DQN_pat5_seed_0_random_False_512000_20250308-192524' # 16×16

    # pretrained_model_name = 'model_Acrobot-v1_DQN_64000_20250301-221607'
    # pretrained_model_name = 'model_Acrobot-v1_DQN_256000_20250301-222022'
    # pretrained_model_name = 'model_MountainCar-v0_DQN_256000_20250301-235147'


if __name__ == '__main__':
    print(Config.learn_kwargs['DQN'])
    print(**Config.learn_kwargs['DQN'])
    