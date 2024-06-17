import sys
import os
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gymnasium
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import helpers.csvReader as csvReader
import modules.mechanics as mech


class GameEnv(gymnasium.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.dungeons = csvReader.load_data_from_folder_RL()
        self.observation_space = spaces.Box(low=0, high=5, shape=(10, 10), dtype=np.uint8)
        # self.state = self.dungeons[random.randint(0, len(self.dungeons) - 1)]
        self.state = self.dungeons[0]
        self.max_actions = 30

    def step(self, action):
        move = {0: mech.UP, 1: mech.RIGHT, 2: mech.DOWN, 3: mech.LEFT}[action]
        tile = mech.lookAhead(self.state, move)
        entity = self.state[tile[0], tile[1]]

        if entity in [mech.WALL, mech.BLOCK, mech.DOOR]:
            reward = 0
        elif entity == mech.ENEMY:
            self.state = mech.killEnemy(self.state, tile)
            reward = 1
        elif entity == mech.EMPTY:
            self.state = mech.iceSliding(self.state, move)
            reward = -1

        self.max_actions -= 1
        done = self.max_actions <= 0
        if mech.isDone(self.state):
            reward = 30
            done = True

        truncated = self.max_actions <= 0  # Nuevo valor para indicar si el episodio se truncó
        info = {}
        return self.state, reward, done, truncated, info  # Devuelve cinco valores

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        # self.state = self.dungeons[random.randint(0, len(self.dungeons) - 1)]
        self.state = self.dungeons[0]
        self.max_actions = 30
        return self.state, {}  # Asegúrate de que el reset también devuelve un diccionario de info

    def render(self, mode='human'):
        print(f"Movimientos: {30 - self.max_actions}\n{self.state}\n")


# MAIN
def main():
    # Crear y entrenar el modelo si es necesario
    env = GameEnv()
    env = DummyVecEnv([lambda: env])

    model = PPO('MlpPolicy', env, learning_rate=1e-3, verbose=1)
    model.learn(total_timesteps=5000,progress_bar=True)
    model.save("ppo_game_model")

    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    # print(f"Mean reward: {mean_reward} +/- {std_reward}")
    # model.learn(total_timesteps=15, log_interval=1)

def test_model():
    # Cargar el modelo guardado
    loaded_model = PPO.load("ppo_game_model")

    # Crear el entorno
    env = GameEnv()

    # Probar el modelo cargado
    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        action, _ = loaded_model.predict(state, deterministic=True)
        state, reward, done, truncated, info = env.step(int(action))
        env.render()

main()
test_model()