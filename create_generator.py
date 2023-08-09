import tensorflow as tf
from keras import layers
import tf_gan

BATCH_SIZE = 32
TRAIN_BUF = 2000
LR = 1e-4

def create_generator():
    model = tf.keras.Sequential()
    
    # creating Dense layer with units 7*7*32(batch_size) and input_shape of (100,)
    model.add(layers.Dense(7*7*BATCH_SIZE, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, BATCH_SIZE)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model
