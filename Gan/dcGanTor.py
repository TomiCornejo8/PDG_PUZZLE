import torch
import numpy as np
from utils import ganColorRenderTor as color
from Gan import ModelTor as modelDc
import os
import torch
import numpy as np

# Limitar el uso de la GPU al 50%
gpu_memory_fraction = 0.8
# Obtén el ID de la GPU (0 si solo tienes una GPU)
gpu_id = 0
# Configura la GPU para utilizar solo un porcentaje específico de su memoria
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=gpu_id)

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

def build_dcgan(generator, discriminator):
    return generator, discriminator

def get_gan(neuronsG,neuronsD, latent_dim, matrixDim, width,height , lrG,lrD):
    channels = matrixDim[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = modelDc.Generator(latent_dim, channels, width, height, neuronsG).to(device)
    discriminator = modelDc.Discriminator(matrixDim, neuronsD).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lrG, betas=(0.5, 0.999))
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=lrD, alpha=0.9, eps=1e-7)
    
    return generator, discriminator, optimizer_g, optimizer_d

def train_dcgan(generator, discriminator, data, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs + 1):
        for _ in range(n_critic):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_imgs = data[idx]
           # real_imgs = np.transpose(real_imgs, (0, 3, 2, 1))
            real_imgs = torch.tensor(real_imgs, dtype=torch.float32).to(device)
            noise = torch.randn((batch_size, latent_dim)).to(device)
            gen_imgs = generator(noise)
            gp = gradient_penalty(discriminator, real_imgs, gen_imgs)

            optimizer_d.zero_grad()
            d_loss_real = discriminator(real_imgs)
            d_loss_fake = discriminator(gen_imgs)
            d_loss = torch.mean(d_loss_fake) - torch.mean(d_loss_real) + 10 * gp
            d_loss.backward()
            optimizer_d.step()

        noise = torch.randn((batch_size, latent_dim)).to(device)
        optimizer_g.zero_grad()
        g_loss = -torch.mean(discriminator(generator(noise)))
        g_loss.backward()
        optimizer_g.step()

        print(f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        if epoch % 100 == 0:
            color.save_images(epoch, generator, discriminator, latent_dim)
