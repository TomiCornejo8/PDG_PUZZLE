# Carpeta donde se encuentran los archivos CSV
from utils import csvReader

import Dcgan as dc
import dcGanpyTorch as dcPy



# Lee todos los archivos CSV en la carpeta y convierte cada uno en un array de 10x10


arrays_list = csvReader.load_data_from_folder()
epochs, batch_size, latent_dim = 10000, 32, 36
gan,generator,discriminator = dc.getGan(latent_dim)
dc.train_dcgan(generator, discriminator, gan, arrays_list, epochs, batch_size, latent_dim) 

""" gan = dcPy.get_gan(latent_dim)
dcPy.train_dcgan(gan.generator,gan.discriminator, arrays_list, epochs, batch_size, gan.latent_dim) """





