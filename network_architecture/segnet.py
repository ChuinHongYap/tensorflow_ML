import tensorflow as tf


def encoder_block(inputs, filters, kernel_size, pool_size):
    conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.layers.Activation('relu')(bn1)
    conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(relu1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    relu2 = tf.keras.layers.Activation('relu')(bn2)
    pool = tf.keras.layers.MaxPooling2D(pool_size)(relu2)
    return pool, conv2


def decoder_block(inputs, skip_features, filters, kernel_size):
    upsample = tf.keras.layers.UpSampling2D((2, 2))(inputs)
    concat = tf.keras.layers.Concatenate()([upsample, skip_features])
    conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(concat)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.layers.Activation('relu')(bn1)
    conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(relu1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    relu2 = tf.keras.layers.Activation('relu')(bn2)
    return relu2


def SegNet(input_shape, n_classes, filters, kernel_size=3, pool_size=(2, 2), n_layers=4):
    # Encoder
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    skip_features = []
    for i in range(n_layers):
        x, features = encoder_block(x, filters[i], kernel_size, pool_size)
        skip_features.append(features)

    # Decoder
    for i in range(n_layers):
        x = decoder_block(x, skip_features[-i-1], filters[-i-1], kernel_size)

    outputs = tf.keras.layers.Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

    # Create model
    model = tf.keras.models.Model(inputs, outputs, name='SegNet')
    return model
