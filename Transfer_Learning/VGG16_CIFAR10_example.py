import tensorflow as tf

'''
Transfer learning VGG16 using CIFAR10
'''

cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

INPUT_SHAPE = train_images.shape[1:]
BATCH_SIZE = 32
EPOCHS = 20

# preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# import VGG16 model with weight and remove a section of layers
base_model = tf.keras.applications.VGG16(input_shape=INPUT_SHAPE,include_top=False, weights="imagenet")
inputs = base_model.inputs

net = tf.keras.layers.GlobalAveragePooling2D()(base_model.layers[-5].output) # cut and add GAP2D at the end of the network
net = tf.keras.layers.Dense(1000, activation='relu')(net)   # add dense
net = tf.keras.layers.Dropout(0.5)(net)   #add dropout layer
outputs = tf.keras.layers.Dense(10, activation='softmax')(net)   # add dense: no. of classes

# IMPORTANT: Freeze the weight of base model
base_model.trainable = False

model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS
          , validation_data=(test_images, test_labels)
          )