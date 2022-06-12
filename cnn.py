# The ML libary we kinda love (lol, thanks dan)
import tensorflow as tf
# Keras -> Wrapper for TF
from tensorflow import keras
# Layers -> Contains types of layers to use in our NN
from tensorflow.keras import layers

import numpy as np

# some CPU/GPU stuff
tf.config.set_visible_devices([], 'GPU')

# grab mnist dataset
mnist = tf.keras.datasets.mnist

# organize data, 28x28 px images
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape=(28, 28)
shape = [-1, 28, 28, 1]

# separate testing and training data
x_train = x_train.reshape(shape)
x_test = x_test.reshape(shape)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# Define and compile model
model = keras.Sequential(
    [
        layers.Conv2D(64, 3, activation="relu", name="MID", input_shape=(28, 28, 1)),
        #layers.Conv2D(64, 3, activation="relu", name="MID2", input_shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(10
        , activation="softmax", name="OUT")
    ]
)

optim = keras.optimizers.Adam(learning_rate = 0.01)
lossmode='categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optim,
              loss=lossmode,
              metrics=metrics)
              
# train & test model
model.fit(x_train, 
          y_train, 
          batch_size=5000,
          epochs=20,
          validation_data = (x_test, y_test))
          
model.evaluate(x = x_test, y = y_test, batch_size = 5000)

