import torch
import time as T

from utils import csvReaderTor as csvReader
from utils import ganColorRenderTor as ImgColor
from utils import expirements as exp
from torchsummary import summary
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
""" 
inicioG = T.time()
generator, discriminator, optimizer_g,scheduler_g, optimizer_d,scheduler_d = Dcgan.get_gan(neuronsG,neuronsD, 
                                                                   latent_dim, matrixDim,lrG,lrD,n_critic, stepSize)

generator, discriminator,optimizer_g,optimizer_d,device=Dcgan.getWeights(generator, discriminator,optimizer_g,optimizer_d)

summary(generator, (latent_dim,))
summary(discriminator, (matrixDim))
# Entrenar el modelo
#Dcgan.train_dcgan(generator, discriminator, dataSet, epochs, batch_size, latent_dim, optimizer_d, optimizer_g,scheduler_g,scheduler_d,matrixDim, n_critic)

exit()

nMaps=[10,100,1000]
results=[]
for i in range(0,len(nMaps)):
    timeI=[]
    nSols=[]
    nMoves=[]
    means=[]
    solvT=[]
    for j in range(1,6):
        print(f"Inicio experimentos {i+1}.{j} tamaño {nMaps[i]}")
        inicioL = T.time()
        noise = torch.randn((nMaps[i], latent_dim),device=device)
        gen_imgs = generator(noise).to(device)
        auxT,auxS,auxM,solvTime=exp.experiment(gen_imgs,f"{i+1}.{j}",inicioL)
        timeI.extend([auxT-inicioL])
        nSols.extend(auxS)
        nMoves.extend(auxM)
        solvT.extend(solvTime)
        print(f"Experimento {i+1}.{j} finalizado en {auxT-inicioL} segundos")
    results=exp.saveMetricResults(results,i,nMoves,nSols,timeI,solvT)
    exp.saveMetrix(results)
     """

matrix_1 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 5, 0, 0, 0, 0, 0, 0, 4, 1],
    [1, 0, 3, 0, 3, 0, 3, 2, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 3, 0, 2, 3, 3, 2, 0, 1],
    [1, 0, 3, 2, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 2, 3, 0, 0, 0, 0, 1],
    [1, 0, 0, 3, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 3, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

matrix_2 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 3, 1, 1],
    [1, 0, 3, 0, 2, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 2, 0, 0, 5, 1],
    [1, 0, 3, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
    [1, 0, 3, 3, 3, 0, 0, 0, 1, 1],
    [1, 0, 3, 0, 0, 0, 3, 0, 1, 1],
    [1, 0, 4, 0, 3, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

matrix_3 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 2, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 4, 1, 1],
    [1, 0, 5, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


ImgColor.save_img([matrix_1,matrix_2,matrix_3])

