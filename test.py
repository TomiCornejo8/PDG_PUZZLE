# Carpeta donde se encuentran los archivos CSV
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import csvReader
from Gan import Dcgan as dc
from tensorflow.keras.optimizers import Adam,RMSprop

import tensorflow as tf
# Lee todos los archivos CSV en la carpeta y convierte cada uno en un array de 10x10
channels = 6
width = 12
height = 12

layerG = 3
neuronsG = 24

layerD = 3
neuronsD = 120

dataSet = csvReader.load_data_from_folder(channels)
epochs = 1200
batch_size = 36
latent_dim =  24
n_critic = 5
matrixDim= (width,height,channels)
optimizerAdam_g = Adam(
    learning_rate=0.00001,  # Común entre 0.001 y 0.0001
    beta_1=0.5,           # Típicamente 0.9
    epsilon=1e-7,         # Pequeño número para evitar divisiones por cero           # Generalmente 0.0, pero ajustable
    amsgrad=False         # Puede ser True para usar AMSGrad
)

optimizerRmsProp_d =  Adam(
    learning_rate=0.00005,  # Común entre 0.001 y 0.0001
    beta_1=0.5,           # Típicamente 0.9
    epsilon=1e-7,         # Pequeño número para evitar divisiones por cero           # Generalmente 0.0, pero ajustable
    amsgrad=False         # Puede ser True para usar AMSGrad
)

gan, generator, discriminator, optimizer_d, optimizer_g = dc.get_gan(layerG,
                                                                     neuronsG,optimizerAdam_g,layerD,neuronsD,
                                                                     optimizerRmsProp_d,latent_dim,matrixDim)
dc.train_dcgan(generator, discriminator, gan, dataSet, epochs, batch_size, latent_dim, optimizer_d,
                optimizer_g, n_critic,"adam")  


""" gan, generator, discriminator, optimizer_d, optimizer_g = dc.get_gan(layerG,layerResidual,layerAttention,neuronsG,optimizerRmsProp_g,layerD,neuronsD,optimizerRmsProp_d,latent_dim,matrixDim,width,height)
dc.train_dcgan(generator, discriminator, gan, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic,"rmsprop")  """
""" gan = dcPy.get_gan(latent_dim)
dcPy.train_dcgan(gan.generator,gan.discriminator, arrays_list, epochs, batch_size, gan.latent_dim) """





