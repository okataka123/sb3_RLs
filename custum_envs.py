import time
import random 
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

class SimpleBattileShip(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.grid_size = 8
        self.num_ship_kind = 1
        self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, self.num_ship_kind])  # noqa: E501
        self.observation_space = spaces.MultiDiscrete(np.array([[[self.num_ship_kind] * self.grid_size] * self.grid_size] * 2))  # noqa: E501
        self.n_detected = 0
        self.reset()


    def reset(self):
        self.enemy_pos = self.init_enemy_pos()

        # 探索結果除法
        self.state = np.zeros((self.grid_size, self.grid_size, self.num_ship_kind, 2))
        self.state_gui = np.full((self.grid_size, self.grid_size), ' ')
    
        # 陣形の中からランダムな1つの鑑定が見つかった状態からスタートする。
        keys = list(self.enemy_pos.keys())
        random_key = random.choice(keys)
        random_value_str = self.enemy_pos[random_key]
        random_value_no = ord(random_value_str) - ord('A')
        random_x = random_key[0]
        random_y = random_key[1]
        self.state[random_x, random_y, random_value_no, 0] = 1
        self.state_gui[random_x, random_y] = random_value_str
        self.n_detected = 1
        return torch.FloatTensor(self.state)


    def step(self, action):
        '''
        actionを受け取ったときの環境の動作ロジックを記述する。

        Args:
            action [int, int, int]: [x, y, pattern]の形。
        '''
        x = int(action[0])
        y = int(action[1])
        pattern = int(action[2])
        pattern_str = chr(ord('A') + int(pattern))
        print(f'x = {x}, y = {y}, pat = {pattern_str}')
        terminated = False
        truncated = False
        reward = -1

        if self.state[x, y, pattern, 0] == 0 and self.state[x, y, pattern, 1] == 0:
            if (x, y) in self.enemy_pos.keys() and self.enemy_pos[(x, y)] == pattern_str:
                self.state[x, y, pattern, 0] = 1
                self.n_detected += 1
                self.state_gui[x, y] = pattern_str
                reward = 1
            else:
                self.state[x, y, pattern, 1] = 1

        if self.n_detected == self.n_enemy:
            terminated = True    

        return self.state, reward, terminated, truncated, {}
    

    def render(self, mode='human'):
        print('[GUI]')
        print(self.state_gui)
        pass


    def init_enemy_pos(self):
        '''
        initialize enemy position.
        '''
        pattern = 1

        if pattern == 1:
            # field patren 1
            enemy_pos = dict()
            enemy_pos[(4, 2)] = 'A'
            enemy_pos[(4, 5)] = 'A'

        elif pattern == 4:
            # field patren 4
            enemy_pos = dict()
            enemy_pos[(2, 2)] = 'A'
            enemy_pos[(2, 5)] = 'A'
            enemy_pos[(5, 5)] = 'A'
            enemy_pos[(5, 2)] = 'A'

        else:
            raise NotImplementedError

        self.n_enemy = len(enemy_pos)

        return enemy_pos


def test_env():
    terminated = False
    truncated = False

    env = SimpleBattileShip()
    state = env.reset()

    step = 0

    while not (terminated or truncated):
        step += 1
        time.sleep(0.01)

        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        print(f'reward: {reward}, terminated: {terminated}, truncated: {truncated}')

    print('step =', step)


if __name__ == '__main__':
    test_env()