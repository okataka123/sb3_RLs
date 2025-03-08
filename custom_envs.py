import time
import random 
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from config import Field_Config

class SimpleBattileShip(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.grid_size = Field_Config.grid_size
        self.num_ship_kind = Field_Config.num_ship_kind
        self.action_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, self.num_ship_kind])  # noqa: E501
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, self.num_ship_kind, 2), dtype=np.int32)  # noqa: E501
        self.n_detected = 0


    def reset(self, seed=None, options=None):
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
        info = {}
        return torch.FloatTensor(self.state), info


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
        # print(f'x = {x}, y = {y}, pat = {pattern_str}')
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

        if Field_Config.pattern == 1:
            # field patren 1
            if not Field_Config.random_translation:
                enemy_pos = dict()
                enemy_pos[(4, 2)] = 'A'
                enemy_pos[(4, 5)] = 'A'
            else:
                enemy_pos = self.generate_random_pos_pat1()

        elif Field_Config.pattern == 4:
            # field patren 4
            if not Field_Config.random_translation:
                enemy_pos = dict()
                enemy_pos[(2, 2)] = 'A'
                enemy_pos[(2, 5)] = 'A'
                enemy_pos[(5, 5)] = 'A'
                enemy_pos[(5, 2)] = 'A'
            else:
                enemy_pos = self.generate_random_pos_pat4()

        elif Field_Config.pattern == 5:
            # field patren 4
            if not Field_Config.random_translation:
                enemy_pos = dict()
                enemy_pos[(1, 3)] = 'A'
                enemy_pos[(2, 2)] = 'A'; enemy_pos[(2, 4)] = 'A'
                enemy_pos[(3, 1)] = 'A'; enemy_pos[(3, 2)] = 'A'; enemy_pos[(3, 4)] = 'A'; enemy_pos[(3, 5)] = 'A'; 
                enemy_pos[(4, 0)] = 'A'; enemy_pos[(4, 3)] = 'A'; enemy_pos[(4, 6)] = 'A'
                enemy_pos[(5, 1)] = 'A'; enemy_pos[(5, 2)] = 'A'; enemy_pos[(5, 4)] = 'A'; enemy_pos[(5, 5)] = 'A'; 
                enemy_pos[(6, 2)] = 'A'; enemy_pos[(6, 4)] = 'A'
                enemy_pos[(7, 3)] = 'A'

                """
                    [
                        [' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ']
                        [' ' ' ' ' ' 'A' ' ' ' ' ' ' ' ']
                        [' ' ' ' 'A' ' ' 'A' ' ' ' ' ' ']
                        [' ' 'A' 'A' ' ' 'A' 'A' ' ' ' ']
                        ['A' ' ' ' ' 'A' ' ' ' ' 'A' ' ']
                        [' ' 'A' 'A' ' ' 'A' 'A' ' ' ' ']
                        [' ' ' ' 'A' ' ' 'A' ' ' ' ' ' ']
                        [' ' ' ' ' ' 'A' ' ' ' ' ' ' ' ']
                    ]
                    
                """
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        self.n_enemy = len(enemy_pos)
        return enemy_pos


    def generate_random_pos_pat1(self):
        diff_x = 3
        while True:
            row_1 = np.random.randint(0, 8)
            col_1 = np.random.randint(0, 8)
            row_2 = row_1
            col_2 = col_1 + diff_x
            cond = (col_2 < 8)
            if cond:
                enemy_pos = dict()
                enemy_pos[(row_1, col_1)] = 'A'
                enemy_pos[(row_2, col_2)] = 'A'
                print('enemy_pos =', enemy_pos)
                break
        return enemy_pos


    def generate_random_pos_pat4(self):
        diff_x_1 = 3
        diff_y_1 = 0
        diff_x_2 = 0
        diff_y_2 = 3
        diff_x_3 = 3
        diff_y_3 = 3
        while True:
            row_1 = np.random.randint(0, 8)
            col_1 = np.random.randint(0, 8)
            row_2 = row_1 + diff_y_1
            col_2 = col_1 + diff_x_1
            row_3 = row_1 + diff_y_2
            col_3 = col_1 + diff_x_2
            row_4 = row_1 + diff_y_3
            col_4 = col_1 + diff_x_3
            cond = (row_2 < 8) & (col_2 < 8) & (row_3 < 8) & (col_3 < 8) & (row_4 < 8) & (col_4 < 8)  # noqa: E501
            if cond:
                enemy_pos = dict()
                enemy_pos[(row_1, col_1)] = 'A'
                enemy_pos[(row_2, col_2)] = 'A'
                enemy_pos[(row_3, col_3)] = 'A'
                enemy_pos[(row_4, col_4)] = 'A'
                print('enemy_pos =', enemy_pos)
                break
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