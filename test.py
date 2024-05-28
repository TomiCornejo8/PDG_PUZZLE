# Carpeta donde se encuentran los archivos CSV
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import csvReader
import Dcgan as dc
import dcGanpyTorch as dcPy



# Lee todos los archivos CSV en la carpeta y convierte cada uno en un array de 10x10
channels = 2
dataSet = csvReader.load_data_from_folder(channels)
epochs =1000
batch_size = 18
latent_dim =  100
matrixDim= (14,14,channels)
gan, generator, discriminator, optimizer_d, optimizer_g = dc.get_gan(latent_dim,matrixDim)
dc.train_dcgan(generator, discriminator, gan, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic=5)

""" gan = dcPy.get_gan(latent_dim)
dcPy.train_dcgan(gan.generator,gan.discriminator, arrays_list, epochs, batch_size, gan.latent_dim) """





