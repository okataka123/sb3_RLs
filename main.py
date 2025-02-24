import os
import argparse
from config import config
from env_wrapper import create_env


def train(env, algo_name, save=False):
    algo = config.algo_dict[algo_name]
    model = algo('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=config.total_timesteps)
    if save:
        modelfile = f'model_{config.env_name}_{algo_name}_{config.total_timesteps}'
        path = os.path.join('trained_models', modelfile)
        model.save(path)


def inference(env, algo_name):
    modelfile = f'model_{config.env_name}_{algo_name}_{config.total_timesteps}'
    path = os.path.join('trained_models', modelfile)

    algo = config.algo_dict[algo_name]
    model = algo.load(path)
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

    if args.mode == 'training':
        print('start training...')
        env = create_env(config.env_name)
        train(env, args.algo, save=True)

    elif args.mode == 'inference':
        print('start inference...')
        env = create_env(config.env_name, render_mode='human')
        for i in range(100):
            print('i =', i)
            inference(env, args.algo)
    else:
        raise NotImplementedError

    env.close()

if __name__ == '__main__':
    main()

