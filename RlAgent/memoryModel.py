import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box

class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, hidden_size: int = 128):
        super(LSTMFeaturesExtractor, self).__init__(observation_space, hidden_size)

        # El tamaño de la entrada será el tamaño del estado
        self.lstm = nn.LSTM(observation_space.shape[0], hidden_size, batch_first=True)
        self.hidden_size = hidden_size

        # Finaliza con una capa fully connected que convierte el hidden_size en output
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, observations: torch.Tensor):
        batch_size = observations.size(0)
        # Inicializa el hidden state
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(observations.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(observations.device)

        # Reorganiza las observaciones para que puedan ser alimentadas en LSTM
        lstm_out, (h_n, c_n) = self.lstm(observations.unsqueeze(1), (h_0, c_0))
        
        # Aplica fully connected al último hidden state
        return self.fc(lstm_out[:, -1, :])