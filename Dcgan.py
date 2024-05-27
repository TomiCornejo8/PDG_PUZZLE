import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras 
from keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

from utils import ganColorRender as color

# Define el generador
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
    beta = layers.Softmax()(beta)
    
    o = layers.multiply([beta, h])
    o = layers.add([x, o])
    return o

def build_generator(latent_dim, channels):
    generator_input = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(72 * 5 * 5)(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((5, 5, 72))(x)
    
    x = layers.Conv2DTranspose(72, kernel_size=4, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = residual_block(x, 72)
    x = attention_block(x, 72)
    
    x = layers.Conv2DTranspose(36, kernel_size=4, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = residual_block(x, 36)
    
    x = layers.Conv2DTranspose(channels, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

    generator = keras.models.Model(generator_input, x)
    return generator

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        validity_interpolated = discriminator(interpolated, training=True)
    gradients = tape.gradient(validity_interpolated, [interpolated])[0]
    gradients_sqr = tf.square(gradients)
    gradient_penalty = tf.reduce_mean(gradients_sqr)
    return gradient_penalty

# Define el discriminador
def build_discriminator(img_shape):
    discriminator_input = keras.Input(shape=img_shape)
    
    x = layers.Conv2D(36, kernel_size=4, strides=2, padding="same")(discriminator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(72, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(1, activation="sigmoid")(x)
    
    discriminator = keras.models.Model(discriminator_input, x)
    return discriminator

# Define la DCGAN
def build_dcgan(generator, discriminator,latent_dim):
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    return gan

def getGan(latent_dim):
    # Parámetros
    latent_dim = latent_dim
    img_shape = (10, 10,6)  # Tamaño de la imagen generada (10x10 píxeles, 6 canales)
    channels = img_shape[-1]
    # Construye y compila el generador y el discriminador
    generator = build_generator(latent_dim, channels)
    discriminator = build_discriminator(img_shape)
    gan = build_dcgan(generator, discriminator,latent_dim)

    lr_schedule = ExponentialDecay(initial_learning_rate=0.000005, decay_steps=100000, decay_rate=0.96, staircase=True)
    optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5)

    discriminator.compile( loss=wasserstein_loss, optimizer=optimizer)
    gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5), loss="binary_crossentropy")
    return gan,generator,discriminator



def train_dcgan(generator, discriminator, gan, data, epochs, batch_size, latent_dim):
    valid = np.ones((batch_size, 1)) * 0.9  # Suavizado de etiquetas para datos reales
    fake = np.zeros((batch_size, 1)) + 0.1  # Suavizado de etiquetas para datos falsos
    d_loss = 0
    print("Training DCGAN...",valid)
    for epoch in range(epochs):
        # Entrena el discriminador con imágenes reales
        idx = np.random.randint(0, len(data), batch_size)
        real_imgs = np.array([data[i] for i in idx])
        
        # Genera un lote de imágenes falsas
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        # Mueve los ejes para que las dimensiones coincidan con lo esperado por el discriminador
        real_imgs = np.moveaxis(real_imgs,[0, 1, 2, 3], [0,3, 2, 1])
        #gen_imgs = np.moveaxis(gen_imgs,[0, 1, 2, 3], [0,3, 2, 1])
        gp = gradient_penalty(discriminator, real_imgs, gen_imgs)
        #d_loss += 10 * gp  # 10 is the lambda parameter for gradient penalty
        # Entrena el discriminador
        if epoch % 5 == 0:
            d_loss_real = discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake) + 10 * gp 
        
        
        # Entrena el generador
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)
        
        # Imprime el progreso
        print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss[0]}]")
        
        # Guarda las imágenes generadas a intervalos regulares
        if epoch % 50 == 0:
            color.save_images(epoch, generator, latent_dim)
