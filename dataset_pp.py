"""
Dataset Processing Module
   Module that preprocesses and processes Udacity's KITTI and CrowdAI vehicle/non-vehicle datasets
"""

import os
import time, sys
import argparse
import numpy as np
import pandas as pd
import cv2
from sys import exit

def update_progress(progress):
    """
    Displays or updates a console progress bar
    :param progress: Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    :return:
    """

    barLength = 50  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:0.2f}% {2}".format("#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def read_dataset(dataset_csv):
    """
    Reads in dataset information into Pandas DataFrame
    :param dataset: dataset labels CSV file path
    :return ds: Pandas DataFrame containing dataset labels
    """

    # read in the dataset
    print("Reading in dataset at {0}...".format(dataset_csv))
    ds = pd.read_csv(dataset_csv, sep=None, engine='python')

    # check for existence of the required columns
    assert 'Frame' in ds.columns, "column Frame doesn't exist in dataset labels"
    assert 'Label' in ds.columns, "column Label doesn't exist in dataset labels"
    assert 'xmin' in ds.columns, "column xmin doesn't exist in dataset labels"
    assert 'xmax' in ds.columns, "column xmax doesn't exist in dataset labels"
    assert 'ymin' in ds.columns, "column ymin doesn't exist in dataset labels"
    assert 'ymax' in ds.columns, "column ymax doesn't exist in dataset labels"

    # find and prepend directory name to all file names
    data_path = os.path.dirname(dataset_csv) + "/"
    ds.Frame = ds.Frame.apply(lambda f: data_path + f)

    return ds


def write_dataset(dataset, outdir, scale, width, height):
    """
    Writes dataset encapsulated in Pandas DataFrame to output directory, along with new labels.csv
    :param dataset:
    :param outdir:
    :param scale:
    :param width:
    :param height:
    :return:
    """

    print("Writing out dataset to {0}...".format(outdir))

    # create output directory if needed
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # copy over images to output directory
    num_frames = len(dataset.Frame)
    i = 0
    for f in dataset.Frame:
        i += 1

        nf = outdir + "/" + os.path.basename(f)
        # print("{0} -> {1}".format(f, nf))
        update_progress(i / num_frames)

        # read in image
        img = cv2.imread(f)

        # resize images as necessary
        if scale:
            height = int(img.shape[0] * scale)
            width = int(img.shape[1] * scale)

        if height and width:
            newimg = cv2.resize(img, (width, height))
        else:
            newimg = img

        # write out image
        cv2.imwrite(nf, newimg)

    # scale bounding boxes
    if scale:
        dataset.xmin = dataset.xmin.apply(lambda x: int(x * scale))
        dataset.xmax = dataset.xmax.apply(lambda x: int(x * scale))
        dataset.ymin = dataset.ymin.apply(lambda y: int(y * scale))
        dataset.ymax = dataset.ymax.apply(lambda y: int(y * scale))
    if height and width:
        # TODO: uh?
        ...

    # write out CSV file
    dataset.Frame = dataset.Frame.apply(lambda f: os.path.basename(f))
    dataset.to_csv(outdir + '/labels.csv', columns=['Frame', 'Label', 'xmin', 'xmax', 'ymin', 'ymax'], index=False)


def cleanup_dataset(dataset):
    """
    Clean up datasets to match up between KITTI and CrowdAI and only include vehicles
    :param dataset:
    :return dataset:
    """

    # keep only columns named 'Frame', 'Label', 'xmin', 'xmax', 'ymin', 'ymax'
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='^((?!Frame|Label|xmin|xmax|ymin|ymax))')))]

    # convert all labels to lowercase
    dataset.Label = dataset.Label.apply(lambda l: l.lower())

    # include only labels of "car" or "truck"
    dataset = dataset[dataset['Label'].isin(['car', 'truck'])]

    return dataset


def data_generator(dataset, batch_size=128):
    """
    Generator to read in, augment, and pre-process image data and yield in batches of batch_size
    :param dataset:
    :param batch_size:
    :return image, labels:
    """

    # read in images & angles
    while 1:
        # shuffle order of log randomly
        rdl = dataset.sample(frac=1).reset_index(drop=True)

        images = []
        angles = []
        for i in range(batch_size):
            """
            if rdl.camera[i] == 'center':
                img = mpimg.imread(data_path + rdl.center[i].strip())
            elif rdl.camera[i] == 'left':
                img = mpimg.imread(data_path + rdl.left[i].strip())
            elif rdl.camera[i] == 'right':
                img = mpimg.imread(data_path + rdl.right[i].strip())
            else:
                print('ERROR: camera angle "{0}" is not implemented!'.format(rdl.camera[i]))
                exit(1)
            if rdl.flip[i]:
                img = np.fliplr(img)
            # TODO: augment with brightness change
            # TODO: augment with random shadows
            # TODO: augment with random image shifts
            # crop top and bottom off the image to reduce parameter space
            img = img[70:135, :, :]
            images.append(img)
            angles.append(rdl.steering[i])
            """
        yield np.array(images), np.array(angles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KITTI/CrowdAI dataset processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scale', type=float, help='scale images by amount')
    parser.add_argument('--width', type=int, help='set width of output images')
    parser.add_argument('--height', type=int, help='set height of output images')
    requiredArgs = parser.add_argument_group('required arguments')
    requiredArgs.add_argument('--outdir', type=str, default='./', help='output directory for processed dataset')
    requiredArgs.add_argument('datasets', type=str, metavar="INPUTCSV", nargs="+",
                              help='dataset labels CSV - a file (e.g., labels.csv) with pointers to pictures and all labels')
    args = parser.parse_args()

    # check for either scale or W/H but not both
    assert args.scale is None or (args.width is None and args.height is None),\
        "cannot specify both scale and width/height"
    assert (args.width is None and args.height is None) or (args.width is not None and args.height is not None),\
        "must specify both height and width"

    # read in the datasets
    for dataset_csv in args.datasets:
        nds = read_dataset(dataset_csv)
        if 'ds' in locals():
            ds = pd.concat([ds, nds], ignore_index=True)
        else:
            ds = nds

    # clean up combined dataset
    ds = cleanup_dataset(ds)

    # write combined dataset
    write_dataset(ds, args.outdir, args.scale, args.width, args.height)
