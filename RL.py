import random
import numpy as np

from helpers import csvReader
from modules import mechanics as mech

import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import keras
import tensorflow as tf

from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
# import ale_py
# from ale_py import ALEInterface

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# Enviorment del  juego
class GameEnv(Env):
    def __init__(self,max_n=20,max_m=20):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(4)
        # Temperature array
        self.observation_space = Box(low=0, high=5, shape=(max_n,max_m), dtype=np.uint8)
        # Load all maps
        self.dungeons = csvReader.load_data_from_folder_RL()
        # Set start map
        self.state = self.dungeons[random.randint(0,len(self.dungeons)-1)]
        # Set max actions
        self.max_actions = 60
        
    def step(self, action):
        if action == 0: # up movement
            move = mech.UP
        elif action == 1: # right movement
            move = mech.RIGHT
        elif action == 2: # down movement
            move = mech.DOWN
        elif action == 3: # left movement
            move = mech.LEFT

        tile = mech.lookAhead(self.state,move)
        entity = self.state[tile[0],tile[1]]

        if entity == mech.WALL or entity == mech.BLOCK or entity == mech.DOOR:
            reward = -2
        elif entity == mech.ENEMY:
            self.state = mech.killEnemy(self.state,entity)
            reward = 1
        elif entity == mech.EMPTY:
            self.state = mech.iceSliding(self.state,move)
            reward = -1

        self.max_actions -= 1 
        
        # Check if game is done
        if self.max_actions <= 0: 
            done = True
        else:
            done = False

        if mech.isDone(self.state):
            reward = 60
            done = True
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        print(f"Movimientos: {60-self.max_actions}")
    
    def reset(self):
        # Set start map
        self.state = self.dungeons[random.randint(0,len(self.dungeons)-1)]
        # Set max actions
        self.max_actions = 60
        return self.state

# Funciones
def build_model(states_shape, actions):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=states_shape))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(4, activation='linear'))
    model.build(input_shape=(None,) + states_shape)
    print(model.output_shape)

    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

# MAIN

# env  = gym.make("ALE/SpaceInvaders-v5")
# states = env.observation_space
# print(f"{states}")

# tf.config.run_functions_eagerly(True)

env = GameEnv()

max_n = 20
max_m = 20

states_shape = (1,max_n,max_m)
actions = env.action_space.n

model = build_model(states_shape, actions)
# model.predict(np.zeros((states_shape)))

# test_input = np.zeros((1,) + states_shape)  # Crea una entrada de prueba con la forma correcta
# model(test_input)
# print(model.output.shape)
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))
_ = dqn.test(env, nb_episodes=15, visualize=True)