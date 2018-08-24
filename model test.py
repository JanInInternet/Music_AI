# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Reshape
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
from tqdm import tqdm


NAME = "Cats-vs-dogs-CNN"

pickle_in = open("X_music.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_music.pickle","rb")
y = pickle.load(pickle_in)


#dense_layers = [0, 1]
#layer_sizes = [32, 64, 128]
#conv_layers = [1, 2, 3]
#Conv_lists = [5, 6]
#MaxPooling_lists = [2, 3]

#tot_len= len(conv_layers)**len(MaxPooling_lists)
#print(range(int(3000)))


print(X.shape)
NAME = "test_music{}".format(int(time.time()))
print(NAME)

model = Sequential()
#X.shape[1:]
model.add(Conv1D(32, (5), input_shape=(7900000, 2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(32, (5)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))


model.add(Conv1D(32, (5)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Reshape((-1, 2)))

#model.add(Flatten())
model.add(Dense(1))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('softmax', input_shape=(4, 4)))


tensorboard = TensorBoard(log_dir="logs.test/{}".format(NAME), histogram_freq=0, write_graph=False, write_grads=False, write_images=False)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
            batch_size=100,
            epochs=10,
            validation_split=0.3,
            verbose=1,
            callbacks=[tensorboard],
            )

model.save("model.model")
