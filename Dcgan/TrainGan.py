import torch
from utils import ganColorRenderTor as color
from Model import model as modelDc
import os
import numpy as np
from torchsummary import summary

# Función de penalización por diversidad
def diversity_penalty(fake_samples):
    return torch.mean((fake_samples[:-1] - fake_samples[1:]).pow(2).sum(1).mean(1))

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

def get_gan(neuronsG,neuronsD, latent_dim, matrixDim, lrG,lrD,n_critic, stepSize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = modelDc.Generator(latent_dim, matrixDim[0], matrixDim[1], matrixDim[1], neuronsG).to(device)
    discriminator = modelDc.Discriminator(matrixDim, neuronsD).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lrG, betas=(0.5, 0.999))
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=stepSize, gamma=0.1)

    optimizer_d = torch.optim.Adam(generator.parameters(), lr=lrD, betas=(0.5, 0.999))
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=n_critic * stepSize, gamma=0.1)
    #torch.optim.RMSprop(discriminator.parameters(), lr=lrD, alpha=0.9, eps=1e-7)
    return generator, discriminator, optimizer_g,scheduler_g, optimizer_d,scheduler_d

def train_dcgan(generator, discriminator, data, epochs, batch_size, latent_dim,
                 optimizer_d, optimizer_g,scheduler_g,scheduler_d,matrixDim, n_critic=5):
    max_norm = 1.0
    gpu_memory_fraction = 0.7
    torch.backends.cudnn.deterministic = True
# Obtén el ID de la GPU (0 si solo tienes una GPU)
    gpu_id = 0
    # Configura la GPU para utilizar solo un porcentaje específico de su memoria
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU")
        x = torch.tensor([1.0], device=device)
        
    summary(generator, (latent_dim,))
    summary(discriminator, (matrixDim))

    discriminator_gradients = []
    generator_gradients = []
    for epoch in range(epochs + 1):
        for _ in range(n_critic):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_imgs = data[idx]
            #real_imgs = np.transpose(real_imgs, (0, 3, 2, 1))
            real_imgs = torch.tensor(real_imgs, dtype=torch.float32).to(device)
            noise = torch.randn((batch_size, latent_dim),device=device)
            gen_imgs = generator(noise).to(device)
            gp = gradient_penalty(discriminator, real_imgs, gen_imgs)

            optimizer_d.zero_grad()
            d_loss_real = discriminator(real_imgs).to(device)
            d_loss_fake = discriminator(gen_imgs).to(device)
            d_loss = torch.mean(d_loss_fake) - torch.mean(d_loss_real) + 5 * gp
            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)

            discriminator_gradients.append(get_gradients(discriminator))

            optimizer_d.step()
            

        noise = torch.randn((batch_size, latent_dim),device=device)
        optimizer_g.zero_grad()
        gen_imgs = generator(noise).to(device)
        g_loss = -torch.mean(discriminator(gen_imgs))
        g_loss += diversity_penalty(gen_imgs)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm)

        generator_gradients.append(get_gradients(generator))

        optimizer_g.step()

        scheduler_d.step()
        scheduler_g.step()

        print(f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        if epoch % 100 == 0:
            color.plot_gradients(generator_gradients, discriminator_gradients, epoch)
            color.save_images(epoch, generator, discriminator, latent_dim)

def get_gradients(model):
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.norm().item())
    return gradients