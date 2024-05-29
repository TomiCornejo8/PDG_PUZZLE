from tensorflow import keras
from keras import layers
import tensorflow as tf

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
        print(neurons)
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
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    for i in range(layer):
        x = layers.Conv2D(neurons, kernel_size=4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
        neurons = neurons // 2


    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    discriminator = keras.models.Model(input, x)
    discriminator.summary()
    return discriminator

