import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete

from custom_envs import SimpleBattileShip
from config import Config


class DiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(np.prod(env.action_space.nvec))

    def action(self, act):
        """Convert Discrete action back to MultiDiscrete format."""
        return np.array(np.unravel_index(act, self.env.action_space.nvec), dtype=int)



def create_env(env_name, render_mode=None):
    env_list = gym.envs.registry.keys()
    if env_name in env_list:
        env = gym.make(Config.env_name, render_mode=render_mode)
    elif env_name == 'SimpleBattileShip':
        env = SimpleBattileShip()
        env = DiscreteWrapper(env)
    else:
        raise NotImplementedError

    return env


def test():
    env_list = gym.envs.registry.keys()
    for env in sorted(env_list):
        print(env)


if __name__ == '__main__':
    test()
