def build_discriminator(img_shape):
    discriminator_input = keras.Input(shape=img_shape)
    
    x = layers.Conv2D(100, kernel_size=4, strides=2, padding="same")(discriminator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(50, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
 
    x = layers.Conv2D(25, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    discriminator = keras.models.Model(discriminator_input, x)
    return discriminator


def build_generator(latent_dim, channels,width,height):
    generator_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(800 * width * height)(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((width, height, 800))(x)
    
    x = layers.Conv2DTranspose(400, kernel_size=4, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = residual_block(x, 400)
    x = attention_block(x, 400)
    
    x = layers.Conv2DTranspose(200, kernel_size=4, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = residual_block(x, 200)
    x = attention_block(x, 200)

    x = layers.Conv2DTranspose(channels, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

    generator = keras.models.Model(generator_input, x)
    return generator