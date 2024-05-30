import os
import torch
import torch.optim as optim
from utils import csvReader
from Gan import dcGanTor as Dcgan

# Configuración
channels = 6
width = 10
height = 10
layerG = 3
layerResidual = 3
layerAttention = 3
neuronsG = 256
layerD = 9
neuronsD = 256
dataSet = csvReader.load_data_from_folderTor(channels)
epochs = 1200
batch_size = 128
latent_dim = 256
n_critic = 2
matrixDim = (channels, width, height)

# Inicializar los modelos
gan, generator, discriminator, optimizer_d, optimizer_g = Dcgan.get_gan(layerG, layerResidual, layerAttention, neuronsG, None, layerD, neuronsD, None, latent_dim, matrixDim, width, height)

# Optimizadores
optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-7)
optimizer_d = optim.RMSprop(discriminator.parameters(), lr=0.000333, alpha=0.9, eps=1e-7)

# Entrenar el modelo
Dcgan.train_dcgan(generator, discriminator, gan, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic)
