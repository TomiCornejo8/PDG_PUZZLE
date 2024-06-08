import os
import torch
import torch.optim as optim
from utils import csvReaderTor as csvReader
from Gan import dcGanTor as Dcgan 
torch.cuda.empty_cache()


# Configuración
channels = 6
width = 10
height = 10
neuronsG = 72
neuronsD = 64
dataSet = csvReader.load_data_from_folderTor(channels)
epochs = 5000
batch_size = 32
latent_dim = 100
n_critic = 2
matrixDim = (channels, width, height)
lrG=0.00005
lrD=0.00001
# Inicializar los modelos
generator, discriminator, optimizer_d, optimizer_g = Dcgan.get_gan(neuronsG,neuronsD, latent_dim, matrixDim, width, height,lrG,lrD)


# Entrenar el modelo
Dcgan.train_dcgan(generator, discriminator, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic)
