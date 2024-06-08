import numpy as np
import matplotlib.pyplot as plt
import torch

def value_to_color(value):
    if value == 0:
        return [1, 1, 1]  # Empty - Blanco
    elif value == 1:
        return [0.5, 0.5, 0.5]  # Wall - Gris
    elif value == 2:
        return [0.6, 0.4, 0.2]  # Block - Marrón
    elif value == 3:
        return [1, 0.6, 0.6]  # Enemy - Rojo claro
    elif value == 4:
        return [0.6, 1, 0.6]  # Door - Verde claro
    elif value == 5:
        return [0.6, 0.8, 1]  # Player - Azul claro
    else:
        return [0, 0, 0]  # Negro para valores no definidos

# Función para convertir la matriz a una imagen en color
def matrix_to_color_image(matrix):
    color_image = np.zeros((matrix.shape[1], matrix.shape[2], 3))
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            color_image[i, j] = value_to_color(matrix[:, i, j].argmax())
    return color_image

def save_images(epoch, generator, discriminator, latent_dim, examples=10, dim=(2, 5), figsize=(18, 6), optim='adam'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn((examples, latent_dim)).to(device)
    gen_imgs = generator(noise)
    discriminate = discriminator(torch.tensor(gen_imgs, dtype=torch.float32).to(device))
    plt.figure(figsize=figsize)
    trueLabel = 0
    for i in range(examples):
        img = gen_imgs[i].cpu().detach().numpy()
        color_img = matrix_to_color_image(img)
        axes = plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(color_img)
        plt.axis('off')
        if discriminate[i] < 0:
            plt.text(0.5, -0.1, "Prediction: False", transform=axes.transAxes, ha="center", fontsize=14)
        else:
            plt.text(0.5, -0.1, "Prediction: True", transform=axes.transAxes, ha="center", fontsize=14)
            trueLabel += 1

    plt.tight_layout()
    plt.savefig(f"{optim}/gen_img_epoch_{epoch}_TrueLabel_{trueLabel}_Samples_{examples}.png")
    plt.close()