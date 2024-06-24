import torch
import torch.optim as optim
from utils import csvReaderTor as csvReader
import TrainGan as Dcgan 
torch.cuda.empty_cache()


# Configuración
channels = 6
width = 10
height = 10

neuronsG = 66
neuronsD = 8

dataSet = csvReader.load_data_from_folderTor(channels)
stepSize =500
epochs = 3000
batch_size = 120
latent_dim = 36
n_critic = 10
matrixDim = (channels, width, height)
lrG=0.006
lrD=0.004
# Inicializar los modelos
generator, discriminator, optimizer_g,scheduler_g, optimizer_d,scheduler_d = Dcgan.get_gan(neuronsG,neuronsD, 
                                                                   latent_dim, matrixDim,lrG,lrD,n_critic, stepSize)


# Entrenar el modelo
Dcgan.train_dcgan(generator, discriminator, dataSet, epochs, batch_size, latent_dim, 
                  optimizer_d, optimizer_g,scheduler_g,scheduler_d,matrixDim, n_critic)