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
input_shape=(28, 28)

# separate testing and training data
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Define and compile model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(512, activation="relu", name="MID"),
        layers.Dense(128, activation="relu", name="MID2"),
        layers.Dropout(rate=0.60),
        layers.Flatten(),
        layers.Dense(10
        , activation="softmax", name="OUT")
    ]
)

optim = keras.optimizers.Adam(learning_rate = 0.01)
loss='categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optim,
              loss=loss,
              metrics=metrics)

# train & test model
model.fit(x_train, 
          y_train, 
          batch_size=5000,
          epochs=50,
          validation_data = (x_test, y_test))
          
model.evaluate(x = x_test, y = y_test, batch_size=5000)

