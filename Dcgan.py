import numpy as np
import os
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

def build_generator(latent_dim, channels,width,height):
    generator_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(144 * width * height)(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((width, height, 144))(x)
    
    x = layers.Conv2DTranspose(144, kernel_size=4, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = residual_block(x, 144)
    x = attention_block(x, 144)
    
    x = layers.Conv2DTranspose(72, kernel_size=4, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = residual_block(x, 72)
    
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
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_penalty = tf.reduce_mean((gradients_sqr_sum - 1.0) ** 2)
    return gradient_penalty

# Define el discriminador
def build_discriminator(img_shape):
    discriminator_input = keras.Input(shape=img_shape)
    
    x = layers.Conv2D(72, kernel_size=4, strides=2, padding="same", kernel_regularizer=keras.regularizers.l2(0.01))(discriminator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(144, kernel_size=4, strides=2, padding="same", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(288, kernel_size=4, strides=2, padding="same", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    
    discriminator = keras.models.Model(discriminator_input, x)
    return discriminator

# Define la DCGAN
def build_dcgan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    return gan

def get_gan(latent_dim,matrixDim,width,height):
    # Parámetros
    img_shape = matrixDim # Tamaño de la imagen generada (14x14 píxeles, 2 canales)
    channels = img_shape[-1]
    
    # Construye y compila el generador y el discriminador
    generator = build_generator(latent_dim, channels,width,height)
    discriminator = build_discriminator(img_shape)
    gan = build_dcgan(generator, discriminator, latent_dim)

    lr_schedule_d = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=10000, decay_rate=0.96, staircase=True)
    lr_schedule_g = ExponentialDecay(initial_learning_rate=0.00001, decay_steps=10000, decay_rate=0.96, staircase=True)
    optimizer_d = Adam(learning_rate=lr_schedule_d, beta_1=0.5, beta_2=0.9)
    optimizer_g = Adam(learning_rate=lr_schedule_g, beta_1=0.5, beta_2=0.9)

    discriminator.compile(loss=wasserstein_loss, optimizer=optimizer_d)
    gan.compile(optimizer=optimizer_g, loss=wasserstein_loss)
    return gan, generator, discriminator, optimizer_d, optimizer_g

def train_dcgan(generator, discriminator, gan, data, epochs, batch_size, latent_dim, optimizer_d, optimizer_g, n_critic=20):
    valid = -tf.ones((batch_size, 1))
    fake = tf.ones((batch_size, 1))
    
    for epoch in range(epochs):
        for _ in range(n_critic):
            # Entrena el discriminador con imágenes reales
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_imgs = data[idx]
            real_imgs = np.moveaxis(real_imgs,[0, 1, 2, 3], [0,3, 2, 1])
            real_imgs = tf.convert_to_tensor(real_imgs, dtype=tf.float32)
            
            # Genera un lote de imágenes falsas
            noise = tf.random.normal((batch_size, latent_dim))
            gen_imgs = generator(noise, training=False)
            
            # Calcula la penalización del gradiente
            gp = gradient_penalty(discriminator, real_imgs, gen_imgs)
            
            with tf.GradientTape() as tape:
                d_loss_real = discriminator(real_imgs, training=True)
                d_loss_fake = discriminator(gen_imgs, training=True)
                d_loss = 0.5 * (tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)) + 10 * gp
            
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            
            if None not in grads and grads:
                optimizer_d.apply_gradients(zip(grads, discriminator.trainable_variables))

        # Entrena el generador
        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as tape:
            g_loss = gan(noise, training=True)
        
        grads = tape.gradient(g_loss, gan.trainable_variables)
        optimizer_g.apply_gradients(zip(grads, gan.trainable_variables))
        
        # Imprime el progreso
        print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss[0]}]")
        
        # Guarda las imágenes generadas a intervalos regulares
        if epoch % 15 == 0:
            color.save_images(epoch, generator, latent_dim)

            # real_imgs = np.moveaxis(real_imgs,[0, 1, 2, 3], [0,3, 2, 1])