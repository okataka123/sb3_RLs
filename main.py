from cmath import polar
import os
import argparse
import datetime as dt
import numpy as np
import torch
from config import Config, Field_Config, Inference_Config
from env_wrapper import create_env


def train(env, algo_name, device='cpu', save=False):
    now = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    algo = Config.algo_dict[algo_name]

    # tensorboardログ設定
    seed = Config.seed
    if Config.env_name == 'SimpleBattleShip':
        rt = str(Field_Config.random_translation)
        pat = Field_Config.pattern
        log_path = os.path.join('tensorboard_log', f'{Config.env_name}_{algo_name}_pat{pat}_seed_{seed}_random_{rt}_{Config.total_timesteps}_{now}')  # noqa: E501
    else:
        log_path = os.path.join('tensorboard_log', f'{Config.env_name}_{algo_name}_seed_{seed}_{Config.total_timesteps}_{now}')  # noqa: E501

    model = algo(
        env=env, 
        device=device, 
        verbose=1,
        tensorboard_log=log_path,
        **Config.learn_kwargs[algo_name],
    )
    model.learn(total_timesteps=Config.total_timesteps)

    if save:
        if Config.env_name == 'SimpleBattleShip':
            rt = str(Field_Config.random_translation)
            pat = Field_Config.pattern
            seed = Config.seed
            modelfile = f'model_{Config.env_name}_{algo_name}_pat{pat}_seed_{seed}_random_{rt}_{Config.total_timesteps}_{now}'  # noqa: E501
        else:
            modelfile = f'model_{Config.env_name}_{algo_name}_{Config.total_timesteps}_{now}'  # noqa: E501
        path = os.path.join('trained_models', modelfile)
        model.save(path)


def inference(env, algo_name, device='cpu'):
    modelfile = Inference_Config.pretrained_model_name

    if Config.env_name not in modelfile:
        print('[error] The model file name and environment name do not match.')
        assert False
    if algo_name not in modelfile:
        print('[error] The model file name and algo_name do not match.')
        assert False
    
    path = os.path.join('trained_models', modelfile)
    algo = Config.algo_dict[algo_name]
    model = algo.load(path, device=device)
    # state, _ = env.reset()
    np.random.seed(None)

    for i in range(100):
        print('i =', i)
        state, _ = env.reset()
        while True:
            env.render()
            action, _ = model.predict(state, deterministic=True)
            print('action =', action)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                env.render()
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
        inference(env, args.algo, device=device)
    else:
        raise NotImplementedError

    env.close()

if __name__ == '__main__':
    main()

