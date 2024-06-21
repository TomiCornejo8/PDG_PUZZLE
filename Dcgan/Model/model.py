import torch
import torch.nn as nn
import torch.nn.functional as F

# Función para inicializar los pesos
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, stride=strides, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Inicialización de pesos
        self.apply(weights_init)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = self.leaky_relu(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, filters):
        super(AttentionBlock, self).__init__()
        self.f_conv = nn.Conv2d(filters, filters // 4, 1)
        self.g_conv = nn.Conv2d(filters, filters // 4, 1)
        self.h_conv = nn.Conv2d(filters, filters, 1)
        self.beta_conv = nn.Conv2d(filters // 4, filters, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Inicialización de pesos
        self.apply(weights_init)

    def forward(self, x):
        f = self.f_conv(x)
        g = self.g_conv(x)
        h = self.h_conv(x)
        s = f + g
        beta = self.beta_conv(s)
        beta = self.sigmoid(beta)
        o = beta * h
        o = o + x
        return o
    
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=6, width=10, height=10, neurons=128):
        super(Generator, self).__init__()
        self.width = width // 4
        self.height = height // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, neurons * self.width * self.height))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(neurons),
            nn.ConvTranspose2d(neurons, neurons // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(neurons // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            ResidualBlock(neurons // 2),
            AttentionBlock(neurons // 2),
            
            nn.ConvTranspose2d(neurons // 2, neurons // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(neurons // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            ResidualBlock(neurons // 4),
            AttentionBlock(neurons // 4),
            
            nn.Conv2d(neurons // 4, channels, 1, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Inicialización de pesos
        self.apply(weights_init)

    def forward(self, x):
        out = self.l1(x)
        out = out.view(out.shape[0], -1, self.width, self.height)  # Ajuste dinámico de los canales
        # Agregar ruido gaussiano aquí
        noise = torch.randn_like(out) * 0.1  # Ajusta la escala del ruido según sea necesario
        out = out + noise
        img = self.conv_blocks(out)
        return img

class WeightClippingConstraint:
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data = torch.clamp(module.weight.data, -self.clip_value, self.clip_value)

class Discriminator(nn.Module):
    def __init__(self, img_shape=(6, 10, 10), neurons=128, clip_value=0.01):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()

        self.layers.append(nn.Sequential(
            nn.Conv2d(img_shape[0], neurons * 4, 2, stride=2, padding=1),
            nn.BatchNorm2d(neurons * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        ))

        self.layers.append(nn.Sequential(
            nn.Conv2d(neurons * 4, neurons * 8, 2, stride=2, padding=1),
            nn.BatchNorm2d(neurons * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        ))

        self.layers.append(nn.Sequential(
            nn.Conv2d(neurons * 8, neurons * 16, 2, stride=2, padding=1),
            nn.BatchNorm2d(neurons * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        ))

        final_size = neurons * 9 * 16
        self.final_layer = nn.Linear(final_size, 1)
        self.apply(WeightClippingConstraint(clip_value))
        
        # Inicialización de pesos
        self.apply(weights_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.final_layer(x)
        return x
