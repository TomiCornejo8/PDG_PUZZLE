import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from utils import ganColorRender as color
# Define el bloque residual
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu(x)
        return x

# Define el bloque de atención
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.f = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, padding=0)
        self.g = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, padding=0)
        self.h = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        f = self.f(x).view(batch_size, -1, height * width)  # [B, C//8, H*W]
        g = self.g(x).view(batch_size, -1, height * width)  # [B, C//8, H*W]
        h = self.h(x).view(batch_size, -1, height * width)  # [B, C, H*W]
        
        s = torch.bmm(f.permute(0, 2, 1), g)  # [B, H*W, H*W]
        beta = self.softmax(s)  # [B, H*W, H*W]
        
        o = torch.bmm(h, beta.permute(0, 2, 1))  # [B, C, H*W]
        o = o.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        x = x + o
        return x

# Define el generador
class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 72 * 5 * 5)
        self.relu = nn.LeakyReLU(0.2)
        self.reshape = lambda x: x.view(-1, 72, 5, 5)
        self.deconv1 = nn.ConvTranspose2d(72, 72, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(72)
        self.res_block1 = ResidualBlock(72, 72)
        self.att_block = AttentionBlock(72)
        self.deconv2 = nn.ConvTranspose2d(72, 36, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(36)
        self.res_block2 = ResidualBlock(36, 36)
        self.deconv3 = nn.ConvTranspose2d(36, channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_block1(x)
        x = self.att_block(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.res_block2(x)
        x = self.deconv3(x)
        x = self.tanh(x)
        return x

# Define el discriminador
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(img_shape[0], 36, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(36)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(36, 72, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(72)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(72 * (img_shape[1] // 4) * (img_shape[2] // 4), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Define la DCGAN
class DCGAN(nn.Module):
    def __init__(self, generator, discriminator, latent_dim):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def forward(self, x):
        gen_imgs = self.generator(x)
        validity = self.discriminator(gen_imgs)
        return validity

def get_gan(latent_dim):
    img_shape = (6, 10, 10)  # Tamaño de la imagen generada (6 canales, 10x10 píxeles)
    channels = img_shape[0]
    generator = Generator(latent_dim, channels)
    discriminator = Discriminator(img_shape)
    gan = DCGAN(generator, discriminator, latent_dim)
    return gan

def train_dcgan(generator, discriminator, data, epochs, batch_size, latent_dim):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.00001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.000005, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for _ in range(len(data) // batch_size):
            real_imgs = np.array([data[i] for i in np.random.randint(0, len(data), batch_size)])
            real_imgs = torch.tensor(real_imgs, dtype=torch.float32).permute(0, 3, 1, 2)

            valid = torch.ones((batch_size, 1)) * 0.9  # Suavizado de etiquetas para datos reales
            fake = torch.zeros((batch_size, 1)) + 0.1  # Suavizado de etiquetas para datos falsos

            optimizer_d.zero_grad()

            noise = torch.randn(batch_size, latent_dim)
            gen_imgs = generator(noise)

            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            g_loss = criterion(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_g.step()

        print(f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        if epoch % 100 == 0:
            color.save_images(epoch, generator, latent_dim)