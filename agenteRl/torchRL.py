import sys
import os
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import gymnasium
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import helpers.csvReader as csvReader
import modules.mechanics as mech


class GameEnv(gymnasium.Env):
    def __init__(self, max_n=20, max_m=20,seed=69):
        # super(GameEnv, self).__init__()
        random.seed(seed)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=5, shape=(max_n, max_m), dtype=np.uint8)
        self.dungeons = csvReader.load_data_from_folder_RL()
        self.state = self.dungeons[random.randint(0, len(self.dungeons) - 1)]
        self.max_actions = 60

    def step(self, action):
        move = {0: mech.UP, 1: mech.RIGHT, 2: mech.DOWN, 3: mech.LEFT}[action]
        tile = mech.lookAhead(self.state, move)
        entity = self.state[tile[0], tile[1]]

        if entity in [mech.WALL, mech.BLOCK, mech.DOOR]:
            reward = -2
        elif entity == mech.ENEMY:
            self.state = mech.killEnemy(self.state, entity)
            reward = 1
        elif entity == mech.EMPTY:
            self.state = mech.iceSliding(self.state, move)
            reward = -1

        self.max_actions -= 1
        done = self.max_actions <= 0 or mech.isDone(self.state)
        if done and mech.isDone(self.state):
            reward = 60

        info = {}
        return self.state, reward, done, info

    def reset(self):
        # random.seed(seed)
        self.state = self.dungeons[random.randint(0, len(self.dungeons) - 1)]
        self.max_actions = 60
        return self.state

    def render(self, mode='human'):
        print(f"Movimientos: {60 - self.max_actions}")


# MAIN
env = GameEnv()
env = DummyVecEnv([lambda: env])

model = DQN('MlpPolicy', env, learning_rate=1e-3, buffer_size=50000, learning_starts=10, target_update_interval=100, verbose=1)
model.learn(total_timesteps=50000)

mean_reward, std_reward = model.evaluate_policy(model.get_env(), n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
model.learn(total_timesteps=15, log_interval=1)
