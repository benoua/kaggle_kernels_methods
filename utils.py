# -*- coding: UTF-8 -*-
# import the necessary packages
import os
import logging
import numpy as np

################################################################################

def load_data(max_rows = 5000):
    """
        Load the data from the data directory.

        Returns:
            - Xtr : training dataset
            - Xte : testing dataset
            - Ytr : training labels
    """
    # define the different path
    Xtr_path = os.path.join(os.getcwd(), "Data", "Xtr.csv")
    Xte_path = os.path.join(os.getcwd(), "Data", "Xte.csv")
    Ytr_path = os.path.join(os.getcwd(), "Data", "Ytr.csv")

    # load the data
    Xtr = np.genfromtxt(Xtr_path, delimiter=',', max_rows = max_rows)
    Xte = np.genfromtxt(Xte_path, delimiter=',', max_rows = max_rows)
    Ytr = np.loadtxt(Ytr_path, delimiter=',', skiprows=1)

    # trim the useless data
    Xtr = Xtr[:,0:3072]
    Xte = Xte[:,0:3072]
    Ytr = Ytr[:,1].astype(np.double)
    Ytr = Ytr[:max_rows]

    return Xtr, Ytr, Xte

################################################################################

def build_image(im_line):
    """
        Convert an image from inline representation to matrix BGR.

        Arguments:
            - im_line : raw 1D image

        Returns:
            - im : raw 2D image
    """
    # retrieve the color channels
    im_R = im_line[0:1024].reshape(32,32)
    im_G = im_line[1024:2048].reshape(32,32)
    im_B = im_line[2048:3072].reshape(32,32)

    # build the final image
    im = np.zeros((32,32,3))
    im[:,:,0] = im_R
    im[:,:,1] = im_G
    im[:,:,2] = im_B

    return im

################################################################################

def raw2gray(im):
    """
        Convert a raw image to its grayscale image between 0 and 1.

        Arguments:
            - im : raw 2D image

        Returns:
            - im_gray : grayscale 2D image
    """
    # average all RGB channels
    im_gray = im.mean(axis=2)

    # normalize the image between 0 and 1
    im_gray = (im_gray - im_gray.min()) / (im_gray.max() - im_gray.min())

    return im_gray



################################################################################

def error(Ypred, Y):
    """
        Compute the error between prediction and ground-truth.

        Arguments:
            - Ypred : predicted data labels
            - Y : data labels

        Returns:
            - err : percentage of false labeling
    """

    return 100. * np.sum(Ypred != Y) / Y.shape[0]

################################################################################

def save_submit(Y):
    """
        Save the submission results as a csv file.

        Arguments:
            - Y : data labels
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # retrieve dimension
    n = Y.shape[0]

    # add the ID
    data = np.hstack((np.arange(1,n+1).reshape(n,1),Y.reshape(n,1)))

    # save to file
    filename = "Yte.csv"
    np.savetxt(filename, data, header="Id,Prediction",
        comments="", fmt="%d", delimiter=',')
    logging.info("Submission save in file %s"%filename)

    return
