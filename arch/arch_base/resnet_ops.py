
import tensorflow as tf


def ResBlockUp(inputs, output_channel):
    init_fn = tf.keras.initializers.glorot_uniform()
    init_fn = tf.function(init_fn, autograph=False)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2DTranspose(output_channel, 5, strides=2, kernel_initializer = init_fn, padding = 'SAME')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(output_channel, 3, strides=1, padding = 'SAME')(x)

    skip = tf.keras.layers.Conv2DTranspose(output_channel, 3,strides=2,kernel_initializer = init_fn, padding = 'SAME')(inputs)

    print(x , skip)
    x = x + skip

    return x


def ResBlockDown(inputs, output_channel):
    init_fn = tf.keras.initializers.glorot_uniform()
    init_fn = tf.function(init_fn, autograph=False)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(output_channel, 5, strides=2, kernel_initializer = init_fn, padding = 'SAME')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv2D(output_channel, 3, strides=1, padding = 'SAME')(x)

    skip = tf.keras.layers.Conv2D(output_channel, 3, strides=2, kernel_initializer = init_fn, padding = 'SAME')(inputs)
    x = x + skip

    return x