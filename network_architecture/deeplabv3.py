import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, BatchNormalization, Reshape, GlobalAveragePooling2D, Conv2DTranspose, Input, UpSampling2D

def conv_block(inputs, filters, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = activation(x)
    return x

def atrous_conv_block(inputs, filters, rate, kernel_size=3, padding='same', activation=tf.nn.relu):
    x = Conv2D(filters, kernel_size, dilation_rate=rate, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = activation(x)
    return x

def ASPP_block(inputs, filters, rates=[6, 12, 18], activation=tf.nn.relu):
    conv1x1 = conv_block(inputs, filters=256, kernel_size=1, activation=activation)
    branch_1 = atrous_conv_block(inputs, filters=filters, rate=rates[0], activation=activation)
    branch_2 = atrous_conv_block(inputs, filters=filters, rate=rates[1], activation=activation)
    branch_3 = atrous_conv_block(inputs, filters=filters, rate=rates[2], activation=activation)
    pool = GlobalAveragePooling2D()(inputs)
    pool = Reshape((1,1,filters))(pool)
    pool = conv_block(pool, filters=filters, kernel_size=1, activation=activation)
    pool = UpSampling2D(size=tf.shape(inputs)[1:3])(pool)
    x = Concatenate(axis=-1)([conv1x1, branch_1, branch_2, branch_3, pool])
    x = conv_block(x, filters=filters, kernel_size=1, activation=activation)
    return x

def DeepLabv3plus(input_shape, num_classes, filters=32, num_layers=5):
    inputs = Input(shape=input_shape)

    # Encoder
    x = conv_block(inputs, filters=filters, strides=2)
    for i in range(num_layers-1):
        filters *= 2
        x = conv_block(x, filters=filters, strides=2)
    
    # Atrous Spatial Pyramid Pooling
    x = ASPP_block(x, filters=filters)
    
    # Decoder
    for i in range(num_layers-1):
        filters //= 2
        x = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(x)
        x = Concatenate(axis=-1)([x, conv_block(inputs, filters=filters, kernel_size=1)])
    
    # Final conv layer
    x = Conv2D(num_classes, kernel_size=1)(x)

    # Output
    outputs = UpSampling2D(size=(input_shape[0]//4, input_shape[1]//4))(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model
