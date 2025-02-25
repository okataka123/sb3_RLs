import os
import argparse
import torch
from config import Config, Field_Config
from env_wrapper import create_env


def train(env, algo_name, device='cpu', save=False):
    algo = Config.algo_dict[algo_name]
    model = algo('MlpPolicy', env, device=device, verbose=1)
    model.learn(total_timesteps=Config.total_timesteps)
    if save:
        if Config.env_name == 'SimpleBattleShip':
            rt = str(Field_Config.random_translation)
            pat = Field_Config.pattern
            modelfile = f'model_{Config.env_name}_{algo_name}_pat{pat}_ramdom_{rt}_{Config.total_timesteps}'  # noqa: E501
        else:
            modelfile = f'model_{Config.env_name}_{algo_name}_{Config.total_timesteps}'
        path = os.path.join('trained_models', modelfile)
        model.save(path)


def inference(env, algo_name, device='cpu'):
    if Config.env_name == 'SimpleBattleShip':
        rt = str(Field_Config.random_translation)
        pat = Field_Config.pattern
        modelfile = f'model_{Config.env_name}_{algo_name}_pat{pat}_ramdom_{rt}_{Config.total_timesteps}'  # noqa: E501
    else:
        modelfile = f'model_{Config.env_name}_{algo_name}_{Config.total_timesteps}'
    path = os.path.join('trained_models', modelfile)
    algo = Config.algo_dict[algo_name]
    model = algo.load(path, device=device)
    state, _ = env.reset()

    while True:
        env.render()
        action, _ = model.predict(state, deterministic=True)
        print('action =', action)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print('done')
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('algo')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'training':
        print('start training...')
        env = create_env(Config.env_name)
        train(env, args.algo, device=device, save=True)

    elif args.mode == 'inference':
        print('start inference...')
        env = create_env(Config.env_name, render_mode='human')
        for i in range(100):
            print('i =', i)
            inference(env, args.algo, device=device)
    else:
        raise NotImplementedError

    env.close()

if __name__ == '__main__':
    main()

