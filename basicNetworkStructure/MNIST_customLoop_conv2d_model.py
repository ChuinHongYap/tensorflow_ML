import tensorflow as tf

'''
MNIST example using convolutional layers with custom tf loop
'''

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

INPUT_SHAPE = train_images.shape[1:]
BATCH_SIZE = 128
EPOCHS = 5

# preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# convert into tf dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


def conv2d_model():
    Input = tf.keras.layers.Input(shape=(INPUT_SHAPE+(1,))) #for gray scale image
    net = tf.keras.layers.Conv2D(16, kernel_size=(3,3), strides=(1, 1))(Input)
    net = tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1, 1))(net)
    net = tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1, 1))(net)
    
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    net = tf.keras.layers.Dense(200, activation='relu')(net)
    
    output = tf.keras.layers.Dense(10, activation='sigmoid')(net)
    
    return tf.keras.Model(Input,output)

model = conv2d_model()
model.summary() #show model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

# training loop
for epoch in range(EPOCHS):
    print('Start of epoch %d' % (epoch+1))
    
    # training loop
    for train_images_batch, train_labels_batch in train_dataset:
        with tf.GradientTape() as tape:
            y_pred = model(train_images_batch)
            
            loss = tf.keras.losses.sparse_categorical_crossentropy(train_labels_batch, y_pred)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # validation loop
    for test_images_batch, test_labels_batch in test_dataset:
        y_pred = model(test_images_batch)
        loss_v = tf.keras.losses.sparse_categorical_crossentropy(test_labels_batch, y_pred)
        
    loss_epoch = tf.math.reduce_mean(loss)
    loss_val = tf.math.reduce_mean(loss_v)
    
    print('Epoch: %s loss = %s val_loss = %s' % (epoch+1, loss_epoch.numpy(), loss_val.numpy()))