import tensorflow as tf
from tf.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Input

def encoder_block(inputs, filters, kernel_size, pool_size):
    conv1 = Conv2D(filters, kernel_size, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)
    conv2 = Conv2D(filters, kernel_size, padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(bn2)
    pool = MaxPooling2D(pool_size)(relu2)
    return pool, conv2


def decoder_block(inputs, skip_features, filters, kernel_size):
    upsample = UpSampling2D((2, 2))(inputs)
    concat = Concatenate()([upsample, skip_features])
    conv1 = Conv2D(filters, kernel_size, padding='same')(concat)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)
    conv2 = Conv2D(filters, kernel_size, padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(bn2)
    return relu2


def SegNet(input_shape, n_classes, filters, kernel_size=3, pool_size=(2, 2), n_layers=4):
    # Encoder
    inputs = Input(shape=input_shape)
    x = inputs
    skip_features = []
    for i in range(n_layers):
        x, features = encoder_block(x, filters[i], kernel_size, pool_size)
        skip_features.append(features)

    # Decoder
    for i in range(n_layers):
        x = decoder_block(x, skip_features[-i-1], filters[-i-1], kernel_size)

    outputs = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)

    # Create model
    model = tf.keras.models.Model(inputs, outputs, name='SegNet')
    return model
