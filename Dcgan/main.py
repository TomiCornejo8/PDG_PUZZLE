import torch
from utils import csvReaderTor as csvReader
from utils import ganColorRenderTor as ImgColor
import TrainGan as Dcgan 
import numpy as np
torch.cuda.empty_cache()


# Configuración
channels = 6
width = 10
height = 10

neuronsG = 84
neuronsD = 8

dataSet = csvReader.load_data_from_folderTor(channels)
stepSize =500
epochs = 20000
batch_size = 72
latent_dim = 66
n_critic = 5
matrixDim = (channels, width, height)
lrG=0.0001
lrD=0.00002
# Inicializar los modelos
generator, discriminator, optimizer_g,scheduler_g, optimizer_d,scheduler_d = Dcgan.get_gan(neuronsG,neuronsD, 
                                                                   latent_dim, matrixDim,lrG,lrD,n_critic, stepSize)


# Entrenar el modelo
Dcgan.train_dcgan(generator, discriminator, dataSet, epochs, batch_size, latent_dim, 
                  optimizer_d, optimizer_g,scheduler_g,scheduler_d,matrixDim, n_critic)


""" generator, discriminator,optimizer_g,optimizer_d,device=Dcgan.getWeights(generator, discriminator,optimizer_g,optimizer_d)
noise = torch.randn((1000, latent_dim),device=device)

gen_imgs = generator(noise).to(device)

mDoor= 0
mPlayer= 0

for img in gen_imgs:
    mapa = np.argmax(img.cpu().detach().numpy(), axis=0)


    if np.where(mapa == 5):
        mPlayer+=1
    if np.where(mapa == 4):
        mDoor+=1

print(f'Player: {mPlayer} Door: {mDoor}')
mapardo = gen_imgs[0].cpu().detach().numpy()
ImgColor.save_img(mapardo)
mapardo = ImgColor.fixMap(mapardo)
ImgColor.save_img(mapardo)
 """
