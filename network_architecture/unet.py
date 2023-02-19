import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters, kernel_size=3, activation='relu', padding='same'):
    conv = Conv2D(num_filters, kernel_size, activation=activation, padding=padding, kernel_initializer='he_normal')(inputs)
    conv = Conv2D(num_filters, kernel_size, activation=activation, padding=padding, kernel_initializer='he_normal')(conv)
    return conv

def unet(input_size=(256, 256, 3), num_classes=3, num_filters=[64, 128, 256, 512], dropout=0.5):
    inputs = tf.keras.layers.Input(input_size)

    # Encoding path
    conv_layers = []
    conv = inputs
    for i, nf in enumerate(num_filters):
        conv = conv_block(conv, nf)
        if i < len(num_filters)-1:
            conv_layers.append(conv)
            conv = MaxPooling2D(pool_size=(2, 2))(conv)
    drop = Dropout(dropout)(conv)

    # Decoding path
    for i, nf in reversed(list(enumerate(num_filters[:-1]))):
        up = UpSampling2D(size=(2, 2))(drop)
        concat = Concatenate(axis=3)([conv_layers[i], up])
        drop = conv_block(concat, nf)

    # Output
    outputs = Conv2D(num_classes, 1, activation='softmax')(drop)

    model = Model(inputs=inputs, outputs=outputs)

    return model
