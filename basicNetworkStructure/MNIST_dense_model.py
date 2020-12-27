import tensorflow as tf

'''
MNIST example using fully connected layers
'''

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

INPUT_SHAPE = train_images.shape[1:]
BATCH_SIZE = 32
EPOCHS = 5

# preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

def simple_model():
  input = tf.keras.layers.Input(shape=(INPUT_SHAPE+(1,))) #for gray scale image
  net = tf.keras.layers.Flatten()(input)  #flatten data
  net = tf.keras.layers.Dense(200, activation='relu')(net)
  net = tf.keras.layers.Dense(200, activation='relu')(net)

  output = tf.keras.layers.Dense(10, activation='sigmoid')(net)

  return tf.keras.Model(input,output)

model = simple_model()
model.summary() #show model

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS
          , validation_data=(test_images, test_labels)
          )
