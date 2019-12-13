import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

class BiGenerator(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(BiGenerator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def call(self, latent, real):
        generated_out = self.decoder(latent)
        encoded_out = self.encoder(real)
        
        return encoded_out, generated_out

    def generate_image(self, latent):
        return self.decoder(latent)


class BiDiscriminator(tf.keras.Model):
    def __init__(self, encoder, hidden_size):
        super(BiDiscriminator, self).__init__()

        self.encoder = encoder

        self.hidden_size = hidden_size
        self.hidden = tf.keras.layers.Dense(hidden_size)

        self.head = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, latent, real):
        real_encoded = self.encoder(real)

        concat = tf.concat((latent, real_encoded), 1)

        hidden = self.hidden(concat)

        out = self.head(hidden)

        return out


def discriminator_loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

        return real_loss + fake_loss


def generator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return real_loss + fake_loss


@tf.function
def train_step(gen, disc, gen_optimizer, disc_optimizer, images):
    noise = tf.random.normal([images.shape[0], 7 * 7 * 128])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_latent, gen_images = gen(noise, images)

        fake_output = disc(noise, gen_images)
        real_output = disc(gen_latent, images)

        gen_loss = generator_loss(real_output, fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, gen.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, disc.trainable_variables))


def train(dataset, gen, disc, gen_optimizer, disc_optimizer, epochs):
    test_img = tf.random.normal([1, 7*7*128])
    total_steps = train_images.shape[0]
    for epoch in range(epochs):

        start = time.time()

        progress = 0
        for i, image_batch in enumerate(dataset):
            train_step(gen, disc, gen_optimizer, disc_optimizer, image_batch)

            if i % 10 == 0:
                plt.imshow(gen.generate_image(test_img)[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
                plt.savefig("C:/Users/mheemske/OneDrive - Capgemini/Documents/Python Projects/BISAGAN/" + f"image_{epoch:02d}_{i:04d}.png")
                plt.axis("off")
                plt.close()

            progress += image_batch.shape[0]
            time_elapsed = time.time() - start
            eta = ((total_steps - progress) / progress) * time_elapsed
            print(f"Steps: {progress} out of {total_steps}. Estimated time left: {eta:.2f} sec." + ' '*4, end='\r')

        display.clear_output(wait=True)

        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Time for epoch {epoch} is {time.time() - start:.2f} sec." + ' '*32)

    display.clear_output(wait=True)


def get_encoder(input_shape):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten()
    ])

    return encoder


if __name__ == "__main__":
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    train_images = train_images[:30000]

    BUFFER_SIZE = 30000
    BATCH_SIZE = 512

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*128, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

    generator_encoder = get_encoder((28, 28, 1))
    discriminator_encoder = get_encoder((28, 28, 1))

    generator = BiGenerator(generator_encoder, decoder)
    discriminator = BiDiscriminator(discriminator_encoder, 256)

    generator_optimizer = tf.keras.optimizers.Adam(5*10**-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(10**-5)

    # checkpoint_dir = './gan_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    #                                 discriminator_optimizer=discriminator_optimizer,
    #                                 generator=generator,
    #                                 discriminator=discriminator)

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    train(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, EPOCHS)