import collections
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras.layers import Dropout

from tensorflow.keras import layers

model = tf.keras.Sequential()

model.add(layers.Embedding(input_dim=1000, output_dim=64))


model.add(layers.LSTM(128))

model.add(layers.Dense(10))

model.summary()

encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None, ))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)


output, state_h, state_c = layers.LSTM(
    64, return_state=True, name='encoder')(encoder_embedded)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None, ))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)


decoder_output = layers.LSTM(
    64, name='decoder')(decoder_embedded, initial_state=encoder_state)
output = layers.Dense(10)(decoder_output)

model = tf.keras.Model([encoder_input, decoder_input], output)
model.summary()


model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))


model.add(layers.GRU(256, return_sequences=True))

model.add(layers.SimpleRNN(128))
model.add(layers.Dense(10))

model.summary() 

paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)


lstm_layer.reset_states()
batch_size = 64

input_dim = 28

units = 64
output_size = 10  # labels are from 0 to 9


def build_model(allow_cudnn_kernel=True):

  if allow_cudnn_kernel:

    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
  else:
  
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))
  model = tf.keras.models.Sequential([
      lstm_layer,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(output_size)]
  )
  return model


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]


model = build_model(allow_cudnn_kernel=True)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer='sgd',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)

slow_model = build_model(allow_cudnn_kernel=False)
slow_model.set_weights(model.get_weights())
slow_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                   optimizer='sgd', 
                   metrics=['accuracy'])
slow_model.fit(x_train, y_train, 
               validation_data=(x_test, y_test), 
               batch_size=batch_size,
               epochs=1)  # We only train for one epoch because it's slower.

#with tf.device('CPU:0'):
#  cpu_model = build_model(allow_cudnn_kernel=True)
#  cpu_model.set_weights(model.get_weights())
#  result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
#  print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
#  plt.imshow(sample, cmap=plt.get_cmap('gray'))               

          