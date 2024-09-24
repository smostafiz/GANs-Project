from keras.datasets.cifar10 import load_data
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose, Conv2D, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

"""Load CIFAR-10 data."""
(X_train, y_train), (_, _) = load_data()

"""Select a single class of images (e.g., ships)."""
X_train = X_train[y_train.flatten() == 8]

"""Plot some images from the training dataset."""
for i in range(49):
    plt.subplot(7, 7, 1 + i)
    plt.axis('off')
    plt.imshow(X_train[i])
plt.show()

"""Define input shape and latent dimension for the GAN."""
img_shape = (32, 32, 3)
latent_dim = 100


def build_generator():
    """This function defines the generator model architecture."""
    model = Sequential()
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


def build_discriminator():
    """This function defines the discriminator model architecture."""
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


"""Build and compile the discriminator model."""
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      metrics=['accuracy'])
discriminator.summary()
plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

"""Build the generator model."""
generator = build_generator()
generator.summary()
plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

"""The generator takes noise as input and generates images."""
z = Input(shape=(latent_dim,))
img = generator(z)

"""For the combined model, we will only train the generator."""
discriminator.trainable = False

"""The discriminator takes generated images as input and determines validity."""
valid = discriminator(img)

"""The combined model (stacked generator and discriminator), and trains the generator to fool the discriminator."""
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))


def show_imgs(epoch):
    """This function defines a function to display generated images."""
    r, c = 4, 4
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    """Rescale images to [0, 1]."""
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, ])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()
    plt.close()


def show_losses(losses):
    """This function defines to display losses."""
    losses = np.array(losses)

    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()


"""Set the number of training epochs, batch size, and display interval."""
epochs = 15000
batch_size = 128
display_interval = 2500
losses = []

"""Normalize the input data."""
X_train = X_train / 127.5 - 1.

"""Adversarial ground truths."""
valid = np.ones((batch_size, 1))
valid += 0.05 * np.random.random(valid.shape)
fake = np.zeros((batch_size, 1))
fake += 0.05 * np.random.random(fake.shape)

"""Main training loop for the specified number of epochs."""
for epoch in range(epochs):
    """Train the discriminator."""
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    """Train the generator."""
    g_loss = combined.train_on_batch(noise, valid)

    """Print and record losses."""
    if epoch % 2500 == 0:
        print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
    if epoch % 1000 == 0:
        losses.append((d_loss[0], g_loss))

    """Display generated images at specified intervals."""
    if epoch % display_interval == 0:
        show_imgs(epoch)

"""Display the generator and discriminator losses over training."""
show_losses(losses)

"""Show a grid of real images from the dataset."""
s = X_train[:40]
s = 0.5 * s + 0.5
f, ax = plt.subplots(5, 8, figsize=(16, 10))
for i, img in enumerate(s):
    ax[i // 8, i % 8].imshow(img)
    ax[i // 8, i % 8].axis('off')
plt.show()

"""Generate and show a grid of images created by the generator."""
noise = np.random.normal(size=(40, latent_dim))
generated_images = generator.predict(noise)
generated_images = 0.5 * generated_images + 0.5
f, ax = plt.subplots(5, 8, figsize=(16, 10))
for i, img in enumerate(generated_images):
    ax[i // 8, i % 8].imshow(img)
    ax[i // 8, i % 8].axis('off')
plt.show()
