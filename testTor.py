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
neuronsG = 144
neuronsD = 6
dataSet = csvReader.load_data_from_folderTor(channels)
epochs = 5000
batch_size = 36
latent_dim = 96
n_critic = 10
matrixDim = (channels, width, height)
lrG=0.00009
lrD=0.00001
# Inicializar los modelos
generator, discriminator, optimizer_g,scheduler_g, optimizer_d,scheduler_d = Dcgan.get_gan(neuronsG,neuronsD, 
                                                                   latent_dim, matrixDim,lrG,lrD)


# Entrenar el modelo
Dcgan.train_dcgan(generator, discriminator, dataSet, epochs, batch_size, latent_dim, 
                  optimizer_d, optimizer_g,scheduler_g,scheduler_d,matrixDim, n_critic)
