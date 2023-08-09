import tensorflow as tf
from keras import layers
import time
import create_generator
import create_discriminator

BATCH_SIZE = 32
TRAIN_BUF = 2000
LR = 1e-4

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    labels=None,
    image_size=(96,96),
    interpolation='nearest',
    shuffle=TRAIN_BUF,
    batch_size=BATCH_SIZE
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.take(500).cache('cached').prefetch(buffer_size=AUTOTUNE)

for image_batch in train_dataset:
  print(image_batch.shape)
  break

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def D_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def G_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(LR)
discriminator_optimizer = tf.keras.optimizers.Adam(LR)
noise_dim = 100
num_of_generated_examples = 16

seed = tf.random.normal([num_of_generated_examples, noise_dim])
generator = create_generator()
discriminator = create_discriminator()

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = G_loss(fake_output)
        disc_loss = D_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train_GAN(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train_GAN(train_dataset, 5)