"""
SSD Model & Trainer
   Module that contains an SSD model, training routines, and model save code
"""

import argparse
import json
import numpy as np
import pandas as pd
from os.path import dirname
from sys import exit
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, ELU, Dropout
from keras.utils import plot_model
from dataset_pp import read_dataset, data_generator


# PARAMETERS FOR US TO SET (can also be set via command line flags)
# batch size to use in generator
BATCH_SIZE = 256
# total samples per epoch
EPOCH_SIZE = 25600
# number of epochs to run
EPOCHS = 5
# should we show the plots or not (e.g., turn off for non-interactive use)
SHOW_PLOTS = False


def ssd_model(input_shape=(160, 320, 3)):
    """Steering model from NVIDIA paper at https://arxiv.org/pdf/1604.07316.pdf"""

    # had to take a guess at activation functions, optimizer, and loss function
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) * 2 - 1, input_shape=input_shape))
    model.add(Conv2D(24,s (5, 5), strides= (2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides= (2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides= (1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides= (1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def train(model, dataset, batch_size=BATCH_SIZE, epoch_size=EPOCH_SIZE, epochs=EPOCHS):
    """Train "epoch_size" samples for "epoch" epochs with a batch size of "batch_size" on model "modelname"."""

    # print model summary
    model.summary()

    # TODO: implement early stopping callback
    # TODO: implement model checkpointing when validation loss improves

    # train model
    print("Training model...")
    trainer = data_generator(dataset, batch_size=batch_size)
    validator = data_generator(dataset, batch_size=batch_size)
    steps_per_epoch = epoch_size / batch_size
    validation_steps = 0.2 * epoch_size / batch_size
    hist = model.fit_generator(trainer, steps_per_epoch=steps_per_epoch, epochs=epochs,
                               validation_data=validator, validation_steps=validation_steps)

    return hist


def plot_training_history(hist, filename=''):
    """Plot histogram of model history"""

    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    if filename:
        plt.savefig(filename)
    if args.showplots:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSD model trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs')
    parser.add_argument('--epochsize', type=int, default=EPOCH_SIZE, help='how many frames per epoch')
    parser.add_argument('--showplots', type=bool, default=SHOW_PLOTS, help='show plots in X')
    requiredArgs = parser.add_argument_group('required arguments')
    requiredArgs.add_argument('dataset', type=str, metavar="CSV", nargs="+",
                              help='dataset labels CSV - a file (e.g., labels.csv) with pointers to pictures and all labels')
    args = parser.parse_args()

    # read in the datasets
    for dataset_csv in args.dataset:
        nds = read_dataset(dataset_csv)
        if 'ds' in locals():
            ds = pd.concat([ds, nds], ignore_index=True)
        else:
            ds = nds

    # create model
    model = ssd_model(input_shape=(65, 320, 3))

    # train the model and plot training history
    history = train(model, ds, batch_size=args.batch, epoch_size=args.epochsize, epochs=args.epochs)
    plot_training_history(history, filename="loss.png")

    # save model
    print("Saving model...")
    # file names
    model_plot = "ssd_model.png"
    model_weights = "ssd_model.h5"
    model_keras = "ssd_model.json"
    # save graph of model to file
    plot_model(model, to_file=model_plot, show_shapes=True, show_layer_names=False)
    # save model weights
    model.save(model_weights)
    # save Keras model
    with open(model_keras, 'w') as outfile:
        json.dump(model.to_json(), outfile)
