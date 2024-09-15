import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cvsReader as reader
import utilFunc as util
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import memoryModel as memory
import torch

class DungeonEnv(gym.Env):
    def __init__(self):
        super(DungeonEnv, self).__init__()
        self.size = 10
        self.action_space = spaces.Discrete(4)  # 4 direcciones
        self.observation_space = spaces.Box(low=0, high=5, shape=(self.size * self.size,), dtype=np.int64)
        self.maps = reader.load_data_from_folder()
        self.dungeon = np.copy(self.maps[np.random.randint(len(self.maps))])
        self.reward= 100
        self.moves= 0
        self.rewardMultiplier= 1
        self.lastMove = None
        self.lastEntity= None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Compatibilidad con gymnasium
        self.dungeon = np.copy(self.maps[np.random.randint(len(self.maps))])
        self.reward= 100
        self.moves= 0
        self.rewardMultiplier = 1
        self.lastMove = None
        self.lastEntity= None
        # Retorna el estado inicial y alguna información adicional
        return self.dungeon.flatten(), {}

    def step_dungeon(self, move):
        tile, entity = util.lookAhead(self.dungeon, move)
        reward = 0
        necessary_enemy = False

        if entity == util.EMPTY:
            self.dungeon = util.iceSliding(self.dungeon, move)
        elif entity == util.ENEMY:
            if util.is_enemy_blocking_path(self.dungeon, tile):
                necessary_enemy = True
                reward += 50  # Recompensa mayor por eliminar un enemigo necesario
            else:
                reward += 10  # Recompensa menor por eliminar un enemigo no necesario
            self.dungeon = util.killEnemy(self.dungeon, tile)
        return self.dungeon, reward, necessary_enemy

    def step(self, action):
        self.dungeon = self.dungeon.reshape((self.size, self.size))
        move = util.MOVES[action]
        tile, entity = util.lookAhead(np.copy(self.dungeon), move)
        done = False
        necessary_enemy = False

        if self.lastMove == move and (self.lastEntity == util.BLOCK or self.lastEntity ==util. WALL):
            self.reward -= 10

        # Llamada a la función step_dungeon
        self.dungeon, reward, necessary_enemy = self.step_dungeon(move)
        self.moves += 1
        # Ajustes en la recompensa
        if necessary_enemy:
            self.reward += reward
        else:
            self.reward += reward

        if util.isDoorNearBy(np.copy(self.dungeon)) and util.getnEnemys(self.dungeon) == 0:
            self.moves += 1
            self.reward += (200 * self.rewardMultiplier)
        
        if util.win(np.copy(self.dungeon)) and util.getnEnemys(self.dungeon) == 0:
            done = True
            self.reward += 200
            return self.dungeon.flatten(), self.reward, done, False, {}
        
        if self.moves > 100:  # Límite de movimientos
            done = True
            self.reward -= 50

        return self.dungeon.flatten(), self.reward, done, False, {}


    def render(self):
        print(self.dungeon)
        print("Moves:", self.moves)
        print("Reward:", self.reward)



def createModel(mainenv):
    print("GPU disponible:", torch.cuda.is_available())
    check_env(mainenv)

    policy_kwargs = dict(
        features_extractor_class=memory.LSTMFeaturesExtractor,
        features_extractor_kwargs=dict(hidden_size=128)  # Tamaño de la capa LSTM
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(
        "MlpPolicy",
        mainenv,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.2,  # Mayor coeficiente de entropía para fomentar exploración
        clip_range=0.2,
        verbose=1
        ,
        device=device
    )

    return model

def getModel(mainenv):
    return PPO.load("ppo_dungeon",env=mainenv)
