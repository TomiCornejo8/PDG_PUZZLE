import torch
import time as T

from utils import csvReaderTor as csvReader
from utils import ganColorRenderTor as ImgColor
from utils import expirements as exp

import TrainGan as Dcgan 
import numpy as np
torch.cuda.empty_cache()


# Configuración
channels = 6
width = 10
height = 10

neuronsG = 84
neuronsD = 8

#dataSet = csvReader.load_data_from_folderTor(channels)
stepSize =500
epochs = 20000
batch_size = 72
latent_dim = 66
n_critic = 5
matrixDim = (channels, width, height)
lrG=0.0001
lrD=0.00002
# Inicializar los modelos
inicioG = T.time()
generator, discriminator, optimizer_g,scheduler_g, optimizer_d,scheduler_d = Dcgan.get_gan(neuronsG,neuronsD, 
                                                                   latent_dim, matrixDim,lrG,lrD,n_critic, stepSize)

generator, discriminator,optimizer_g,optimizer_d,device=Dcgan.getWeights(generator, discriminator,optimizer_g,optimizer_d)
# Entrenar el modelo
#Dcgan.train_dcgan(generator, discriminator, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g,scheduler_g,scheduler_d,matrixDim, n_critic)



nMaps=[10,10,100,100,200,200,1000]

for i in range(0,len(nMaps)-3):
    print("Inicio experimentos")
    inicioL = T.time()
    noise = torch.randn((nMaps[i], latent_dim),device=device)
    gen_imgs = generator(noise).to(device)
    exp.experiment(gen_imgs,i+1)
    finL=T.time()
    print(f"Experimento {i+1} finalizado en {finL-inicioL} segundos")

