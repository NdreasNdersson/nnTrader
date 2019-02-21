import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import Nadam

from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)


def run_neural_network(x_train, y_train, x_test, y_test):

    # Building the Artificial Neural Network
    model = Sequential()
    model.add(Convolution1D(input_shape=(x_train.shape[1], x_train.shape[2]),
                            nb_filter=16,
                            filter_length=4,
                            border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Convolution1D(nb_filter=8,
                            filter_length=4,
                            border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(2))
    model.add(Activation('softmax'))

    opt = Nadam(lr=0.002)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath="lolkek.hdf5", verbose=1, save_best_only=True)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=100,
                        batch_size=128,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[reduce_lr, checkpointer],
                        shuffle=True)

    model.load_weights("lolkek.hdf5")
    prediction = model.predict(np.array(x_test))

    return history, prediction
