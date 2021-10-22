import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.regularizers import l1

import data_preparator as dp

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import constants as c

# x_train = x_train.astype('float32') / 120.
# x_test = x_test.astype('float32') / 120.

score_parts = dp.get_score_parts()
x = dp.get_input_data(score_parts)
x = x.reshape((-1, c.INPUT_MAX_WIDTH, c.INPUT_MAX_DEPTH, 1)) / 120
np.random.shuffle(x)

x_train = x[0:3000]
x_test = x[3000:3300]

x_vad = x[3300:]

print (x.shape)

latent_dim = 8

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(c.INPUT_MAX_WIDTH, c.INPUT_MAX_DEPTH, 1)),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
      layers.MaxPooling2D((2, 2), padding='same')])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same', strides=2),
      layers.UpSampling2D((2, 2)),
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.UpSampling2D((2, 2)),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    # self.encoder = tf.keras.Sequential([
    #   layers.Dense(16, activation="relu"),
    #   layers.Dense(8, activation="relu")])
    #
    # self.decoder = tf.keras.Sequential([
    #   layers.Dense(16, activation="relu"),
    #   layers.Dense(1, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x_train, x_train,
                epochs=50,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_vad).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy() * 120
aaa = x_vad * 120


print('a')