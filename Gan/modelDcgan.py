import keras
from keras import layers
import tensorflow as tf

def residual_block(x, filters, kernel_size=1, strides=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    return x

def attention_block(x, filters):
    f = layers.Conv2D(filters // 2, 1, padding='same')(x)
    g = layers.Conv2D(filters // 2, 1, padding='same')(x)
    h = layers.Conv2D(filters, 1, padding='same')(x)
    
    s = layers.add([f, g])
    beta = layers.Conv2D(filters, 1, padding='same')(s)
    beta = layers.Activation('sigmoid')(beta)
    
    o = layers.multiply([beta, h])
    o = layers.add([x, o])
    return o

def buildGenerator(layer=2, neurons=100, latent_dim=100, channels=6, width=12, height=12):
    input = tf.keras.Input(shape=(latent_dim,))
    initial_width = width // 6
    initial_height = height // 6
    x = layers.Dense(neurons * initial_width * initial_height)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((initial_height, initial_width,neurons))(x)

    x = layers.Conv2DTranspose(neurons, kernel_size=1, strides=3, padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = residual_block(x, neurons)
    x = attention_block(x, neurons)

    for i in range(layer):
        strides = 2 if i < layer - 1 else 1
        x = layers.Conv2DTranspose(neurons, kernel_size=2, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = residual_block(x, neurons)
        x = attention_block(x, neurons)
        neurons = neurons *2 
    x = layers.Conv2D(neurons, kernel_size=2, strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(channels, kernel_size=4, strides=1, padding="same", activation="sigmoid")(x)

    generator = tf.keras.models.Model(input, x)
    generator.summary()
    return generator
class WeightClippingConstraint(keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, w):
        return keras.backend.clip(w, -self.clip_value, self.clip_value)

# Define el discriminador
def buildDiscriminator(layer=2,neurons=6,img_shape=6):
    clip_constraint = WeightClippingConstraint(clip_value=0.01)
    input = keras.Input(shape=img_shape)
    x = layers.Conv2D(neurons, kernel_size=1, strides=2, padding="valid",kernel_constraint=clip_constraint)(input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)
    neurons = neurons // 2
    for i in range(layer):
        x = layers.Conv2D(neurons, kernel_size=2, strides=2, padding="same",kernel_constraint=clip_constraint)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dropout(0.3)(x)
        neurons = neurons // 2


    x = layers.Flatten()(x)
    x = layers.Dense(1,kernel_constraint=clip_constraint)(x)
    
    discriminator = keras.models.Model(input, x)
    discriminator.summary()
    return discriminator


