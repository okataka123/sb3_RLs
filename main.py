import os
import argparse
import random
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from config import Config, Field_Config, Inference_Config
from env_wrapper import create_env
from util import Util


def train(env, algo_name, device='cpu', save=False):
    now = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    algo = Config.algo_dict[algo_name]

    # tensorboardログ設定
    seed = Config.seed
    if Config.env_name == 'SimpleBattleShip':
        rt = str(Field_Config.random_translation)
        pat = Field_Config.pattern
        log_path = os.path.join('tensorboard_log', f'{Config.env_name}_{algo_name}_l1_{Config.n_node_layer1}_l2_{Config.n_node_layer2}_gridsize_{Field_Config.grid_size}_pat{pat}_seed_{seed}_random_{rt}_{Config.total_timesteps}_{now}')  # noqa: E501
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
            modelfile = f'model_{Config.env_name}_{algo_name}_l1_{Config.n_node_layer1}_l2_{Config.n_node_layer2}_gridsize_{Field_Config.grid_size}_pat{pat}_seed_{seed}_random_{rt}_{Config.total_timesteps}_{now}'  # noqa: E501
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
    np.random.seed(None)
    random.seed(None)

    if Inference_Config.save_Q_fig:
        Util.check_and_clean_directory('Q_figs')

    for i in range(Inference_Config.n_episode):
        print('i =', i+1)
        state, _ = env.reset()
        step = 0
        while True:
            if Inference_Config.save_Q_fig:
                q_value = model.q_net.forward(model.policy.obs_to_tensor(state)[0]).detach().cpu().numpy()
                q_value = q_value.reshape(Field_Config.grid_size, Field_Config.grid_size)
                plt.figure(figsize=(10, 6))
                sns.heatmap(q_value, annot=True, fmt=".3f", cmap="viridis")
                plt.savefig(f'Q_figs/q_val_step_{step}.png')
            print(f'step: {step}')
            env.render()
            action, _ = model.predict(state, deterministic=True)
            print('action =', action)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if done:
                print(f'step: {step}')
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