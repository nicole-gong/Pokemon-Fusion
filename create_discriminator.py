import tensorflow as tf
from keras import layers
import tf_gan

def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(
        64, 
        (5, 5), 
        strides=(2, 2), 
        padding='same', 
        input_shape= [96, 96, 3])
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model