import tensorflow as tf

'''
Compilation of activation layers based on keras layer API
'''

class relu(tf.keras.layers.Layer):
  def __init__(self):
    super(relu, self).__init__()

  def call(self, x):
    return tf.math.maximum(0.,x)


class leakyrelu(tf.keras.layers.Layer):
  def __init__(self,alpha=0.01):
    super(leakyrelu, self).__init__()
    self.alpha=alpha

  def call(self, x):
    return tf.math.maximum(self.alpha*x , x)


class elu(tf.keras.layers.Layer):
  def __init__(self,alpha=1.0):
    super(elu, self).__init__()
    self.alpha=alpha

  def call(self, x):
    return x if x >= 0 else self.alpha*(tf.math.exp(x)-1)


class sigmoid(tf.keras.layers.Layer):
  def __init__(self):
    super(sigmoid, self).__init__()

  def call(self, x):
    return 1/(1+tf.math.exp(-x))


class tanh(tf.keras.layers.Layer):
  def __init__(self):
    super(tanh, self).__init__()

  def call(self, x):
    return tf.math.tanh(x)


class softmax(tf.keras.layers.Layer):
  def __init__(self):
    super(softmax, self).__init__()

  def call(self, x):
    return tf.math.exp(x) / tf.math.reduce_sum(tf.math.exp(x)) 