import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import seaborn as sns

def createModel():
    # -*- coding: utf-8 -*-

    # ~~~1. Data set ~~~



    np.random.seed(2)

    'load the dataset'
    dataset = pd.read_csv("dataset.csv")

    "creating label"
    y = dataset["label"]

    "dropping label"
    X = dataset.drop(labels=["label"], axis=1)

    "deleting dataset to reduce memory usage"
    del dataset

    'overview of dataset'
    g = sns.countplot(y)
    y.value_counts()

    'Grayscale normalization to reduce the effect of illumination differences.'
    X = X / 255.0

    'reshaping the dataset to fit standard of a 4D tensor of shape [mini-batch size, height = 28px, width = 28px, channels = 1 due to grayscale].'
    X = X.values.reshape(-1, 28, 28, 1)

    'categorical conversion of label'
    y = to_categorical(y, num_classes=14)

    '90% Training and 10% Validation split'
    random_seed = 2
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_seed, stratify=y)

    # -------------------------------

    # ~~~2. Model~~~
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
    from keras.optimizers import RMSprop
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ReduceLROnPlateau
    from tensorflow import keras

    'creating the instance of the model'
    model = Sequential()

    'adding layers to the model'
    # Layer: 1
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="Same", activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layer: 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # fully connected layer and output
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(14, activation="softmax"))

    'Set the optimizer and annealer'
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy",
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0001)

    'data augmentation'
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    'fitting the model'
    epochs = 5
    batch_size = 86

    history = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,  # An epoch is an iteration over the entire x and y data provided
        validation_data=(X_val, y_val),
        # Data on which to evaluate the loss and any model metrics at the end of each epoch.
        verbose=2,  # output
        steps_per_epoch=X_train.shape[0] // batch_size,
        # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
        callbacks=[learning_rate_reduction]
    )

    'saving the model HDF5 binary data format'
    model.save("model.h5")