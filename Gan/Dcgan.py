import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay


#Internal 
from utils import ganColorRender as color
from Gan import modelDcgan as modelDc

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


# Define la DCGAN
def build_dcgan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    return gan

def get_gan(layerG,layerResidual,layerAttention,neuronsG,optimizer_g,layerD,neuronsD,optimizer_d,latent_dim,matrixDim,width,height):
    # Parámetros
    img_shape = matrixDim 
    channels = img_shape[-1]

    generator = modelDc.buildGenerator(layerG,layerResidual,layerAttention,neuronsG,latent_dim, channels,width,height)
    discriminator=modelDc.buildDiscriminator(layerD,neuronsD,img_shape)

    gan = build_dcgan(generator, discriminator, latent_dim)

    discriminator.compile(loss= wasserstein_loss, optimizer=optimizer_d)
    gan.compile(optimizer=optimizer_g, loss=wasserstein_loss)
    return gan, generator, discriminator, optimizer_d, optimizer_g

def train_dcgan(generator, discriminator, gan, data, epochs, batch_size, latent_dim, optimizer_d, 
                optimizer_g, n_critic=5,optim=''):

    for epoch in range(epochs + 1):
        for _ in range(n_critic):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_imgs = data[idx]
            real_imgs = np.moveaxis(real_imgs, [0, 1, 2, 3], [0, 3, 2, 1])
            real_imgs = tf.convert_to_tensor(real_imgs, dtype=tf.float32)
            
            noise = tf.random.normal((batch_size, latent_dim))
            gen_imgs = generator(noise, training=False)
            gp = gradient_penalty(discriminator, real_imgs, gen_imgs)
            
            with tf.GradientTape() as tape:
                d_loss_real = discriminator(real_imgs, training=True)
                d_loss_fake = discriminator(gen_imgs, training=True)
                d_loss = tf.reduce_mean(d_loss_fake) - tf.reduce_mean(d_loss_real) + 10 * gp
            
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            
            if None not in grads and grads:
                optimizer_d.apply_gradients(zip(grads, discriminator.trainable_variables))

        noise = tf.random.normal((batch_size , latent_dim))
        with tf.GradientTape() as tape:
            g_loss = -tf.reduce_mean(discriminator(generator(noise, training=True)))
        
        grads = tape.gradient(g_loss, generator.trainable_variables)
        optimizer_g.apply_gradients(zip(grads, generator.trainable_variables))
        
        print(f"{epoch} [D loss: {d_loss.numpy()}] [G loss: {g_loss.numpy()}]")
        
        if epoch % 100 == 0:
            color.save_imagess(epoch=epoch, generator=generator,discriminator=discriminator, latent_dim=latent_dim,optim=optim)