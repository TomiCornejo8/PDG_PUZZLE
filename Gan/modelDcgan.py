import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        shortcut = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = F.leaky_relu(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, filters):
        super(AttentionBlock, self).__init__()
        self.f = nn.Conv2d(filters, filters // 8, kernel_size=1)
        self.g = nn.Conv2d(filters, filters // 8, kernel_size=1)
        self.h = nn.Conv2d(filters, filters, kernel_size=1)
        self.beta = nn.Conv2d(filters, filters, kernel_size=1)

    def forward(self, x):
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        s = f + g
        beta = torch.sigmoid(self.beta(s))
        o = beta * h
        o = o + x
        return o

class Generator(nn.Module):
    def __init__(self, latent_dim, channels, width, height, layer=2, layerResidual=2, layerAttention=2, neurons=100):
        super(Generator, self).__init__()
        self.width = width // 2
        self.height = height // 2
        self.dense = nn.Linear(latent_dim, neurons * self.width * self.height)
        self.bn = nn.BatchNorm1d(neurons * self.width * self.height)
        self.layers = nn.ModuleList()
        for _ in range(layer):
            self.layers.append(nn.ConvTranspose2d(neurons, neurons, kernel_size=4, stride=1, padding=1))
            self.layers.append(nn.BatchNorm2d(neurons))
            self.layers.append(ResidualBlock(neurons) if layerResidual > 0 else nn.Identity())
            self.layers.append(AttentionBlock(neurons) if layerAttention > 0 else nn.Identity())
            neurons //= 2
        self.conv_final = nn.ConvTranspose2d(neurons * 2, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor to 2D for BatchNorm1d
        x = F.leaky_relu(self.bn(x))
        x = x.view(x.size(0), -1, self.height, self.width)  # Reshape the tensor back to 4D for Conv2d
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        x = torch.tanh(self.conv_final(x))
        return x
class Discriminator(nn.Module):
    def __init__(self, channels, layer=2, neurons=100):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(channels, neurons, kernel_size=4, stride=2, padding=1))
        for _ in range(layer):
            self.layers.append(nn.Conv2d(neurons, neurons, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.Dropout(0.3))
            neurons //= 2
        self.fc = nn.Linear(neurons * 4 * 4, 1)  # Ajusta esto según la forma final

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


from tensorflow import keras
from keras import layers


def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.LeakyReLU()(x)
    return x

def attention_block(x, filters):
    f = layers.Conv2D(filters // 8, 1, padding='same')(x)
    g = layers.Conv2D(filters // 8, 1, padding='same')(x)
    h = layers.Conv2D(filters, 1, padding='same')(x)
    
    s = layers.add([f, g])
    beta = layers.Conv2D(filters, 1, padding='same')(s)  # Ensure beta has the same shape as h
    beta = layers.Activation('sigmoid')(beta)
    
    o = layers.multiply([beta, h])
    o = layers.add([x, o])
    return o


def buildGenerator(layer=2,layerResidual = 2,layerAttention = 2,neurons=100,latent_dim=100, channels=6,width=5,height=5):
    input = keras.Input(shape=(latent_dim,))
    width = width // 2  
    height =  height // 2
    x = layers.Dense(neurons * width * height)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((width, height, neurons))(x)

    for _ in range(layer):
        x = layers.Conv2DTranspose(neurons , kernel_size=4, strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        if layerResidual > 0:
                x = residual_block(x, neurons )
                layerResidual -= 1
        if layerAttention > 0:
                x = attention_block(x, neurons )
                layerAttention -= 1
        neurons = neurons // 2

    x = layers.Conv2DTranspose(channels, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

    generator = keras.models.Model(input, x)
    generator.summary()
    return generator

# Define el discriminador
def buildDiscriminator(layer=2,neurons=100,img_shape=6):
    input = keras.Input(shape=img_shape)
    x = layers.Conv2D(neurons, kernel_size=4, strides=2, padding="same")(input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    for _ in range(layer):
        x = layers.Conv2D(neurons, kernel_size=4, strides=2, padding="same")(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        neurons = neurons // 2


    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    discriminator = keras.models.Model(input, x)
    discriminator.summary()
    return discriminator
