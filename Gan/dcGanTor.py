import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from utils import ganColorRender as color
from Gan import modelDcgan as modelDc

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(d_interpolated.size()).to(real_samples.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def build_dcgan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = Variable(torch.FloatTensor(latent_dim))
    gan_output = discriminator(gan_input)
    gan = nn.Sequential(gan_input, gan_output)
    return gan

def get_gan(layerG, layerResidual, layerAttention, neuronsG, optimizer_g, layerD, neuronsD, optimizer_d, latent_dim, matrixDim, width, height):
    channels = matrixDim[0]
    generator = modelDc.Generator(latent_dim, channels, width, height, layerG, layerResidual, layerAttention, neuronsG)
    discriminator = modelDc.Discriminator(channels, layerD, neuronsD)
    gan = build_dcgan(generator, discriminator, latent_dim)
    return gan, generator, discriminator, optimizer_d, optimizer_g

def train_dcgan(generator, discriminator, gan, data, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic=5, optim=''):
    for epoch in range(epochs + 1):
        for _ in range(n_critic):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_imgs = data[idx]
            real_imgs = np.transpose(real_imgs, (0, 3, 2, 1))
            real_imgs = torch.tensor(real_imgs, dtype=torch.float32)

            noise = torch.randn((batch_size, latent_dim))
            gen_imgs = generator(noise)
            gp = gradient_penalty(discriminator, real_imgs, gen_imgs)

            optimizer_d.zero_grad()
            d_loss_real = discriminator(real_imgs)
            d_loss_fake = discriminator(gen_imgs)
            d_loss = torch.mean(d_loss_fake) - torch.mean(d_loss_real) + 10 * gp
            d_loss.backward()
            optimizer_d.step()

        noise = torch.randn((batch_size, latent_dim))
        optimizer_g.zero_grad()
        g_loss = -torch.mean(discriminator(generator(noise)))
        g_loss.backward()
        optimizer_g.step()

        print(f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        if epoch % 100 == 0:
            color.save_images(epoch, generator, discriminator, latent_dim, optim)
