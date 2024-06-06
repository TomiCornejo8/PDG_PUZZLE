# Carpeta donde se encuentran los archivos CSV
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import csvReader
from Gan import Dcgan as dc
import dcGanpyTorch as dcPy
from tensorflow.keras.optimizers import Adam,RMSprop

import tensorflow as tf
# Lee todos los archivos CSV en la carpeta y convierte cada uno en un array de 10x10
channels = 6
width = 10
height = 10

layerG = 2
layerResidual = 2
layerAttention = 2
neuronsG = 128

layerD = 7
neuronsD = 32

dataSet = csvReader.load_data_from_folder(channels)
epochs = 1200
batch_size = 256
latent_dim =  128
n_critic = 10
matrixDim= (width,height,channels)
optimizerAdam_g = Adam(
    learning_rate=0.00005,  # Común entre 0.001 y 0.0001
    beta_1=0.9,           # Típicamente 0.9
    beta_2=0.999,         # Típicamente 0.999
    epsilon=1e-7,         # Pequeño número para evitar divisiones por cero           # Generalmente 0.0, pero ajustable
    amsgrad=False         # Puede ser True para usar AMSGrad
)

optimizerRmsProp_d =  RMSprop(
    learning_rate=0.0001,   # Usualmente entre 0.001 y 0.0001
    rho=0.9,               # Comúnmente 0.9
    momentum=0.0,          # Generalmente 0.0, pero puede ajustarse
    epsilon=1e-7,          # Pequeño número para evitar divisiones por cero
    centered=False         # Puede ser True para normalizar gradientes
)

gan, generator, discriminator, optimizer_d, optimizer_g = dc.get_gan(layerG,layerResidual,layerAttention,
                                                                     neuronsG,optimizerAdam_g,layerD,neuronsD,
                                                                     optimizerRmsProp_d,latent_dim,matrixDim,width,height)
dc.train_dcgan(generator, discriminator, gan, dataSet, epochs, batch_size, latent_dim, optimizer_d,
                optimizer_g, n_critic,"adam")  


""" gan, generator, discriminator, optimizer_d, optimizer_g = dc.get_gan(layerG,layerResidual,layerAttention,neuronsG,optimizerRmsProp_g,layerD,neuronsD,optimizerRmsProp_d,latent_dim,matrixDim,width,height)
dc.train_dcgan(generator, discriminator, gan, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic,"rmsprop")  """
""" gan = dcPy.get_gan(latent_dim)
dcPy.train_dcgan(gan.generator,gan.discriminator, arrays_list, epochs, batch_size, gan.latent_dim) """





