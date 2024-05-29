# Carpeta donde se encuentran los archivos CSV
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import csvReader
from Gan import Dcgan as dc
import dcGanpyTorch as dcPy



# Lee todos los archivos CSV en la carpeta y convierte cada uno en un array de 10x10
channels = 6
width = 10
height = 10

layerG = 3
layerResidual = 2
layerAttention = 2
neuronsG = 256

layerD=  3
neuronsD = 128

dataSet = csvReader.load_data_from_folder(channels,height,width)

epochs =1000
batch_size = 16
latent_dim =  100
n_critic=3
matrixDim= (width,height,channels)
gan, generator, discriminator, optimizer_d, optimizer_g = dc.get_gan(layerG,layerResidual,layerAttention,neuronsG,layerD,neuronsD,latent_dim,matrixDim,width,height)
dc.train_dcgan(generator, discriminator, gan, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic) 

""" gan = dcPy.get_gan(latent_dim)
dcPy.train_dcgan(gan.generator,gan.discriminator, arrays_list, epochs, batch_size, gan.latent_dim) """





