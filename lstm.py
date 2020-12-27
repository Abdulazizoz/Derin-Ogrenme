from __future__ import print_function
#LSTM
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input,Dense,TimeDistributed
from keras.layers import LSTM

batch_size=32
num_classes=10
epochs=5

row_hidden=128
col_hidden=128

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_test=x_test.astype('float32')
x_train=x_train.astype('float32')

x_train/=255
x_test/=255
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

row, col, pixel=x_train.shape[1:]

x=Input(shape=(row, col, pixel))

encoded_rows=TimeDistributed(LSTM(row_hidden))(x)

encoded_columns=LSTM(col_hidden)(encoded_rows)
prediction=Dense(num_classes, activation='softmax')(encoded_columns)
model=Model(x,prediction)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(x_test,y_test))

scores= model.evaluate(x_test,y_test,verbose=0)
print('Test Loss:',scores[0])
print('Test Accuracy:',scores[1])

