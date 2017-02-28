# -*- coding: UTF-8 -*-
# import the necessary packages
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import cvxopt
from sklearn import svm
import sys
import pdb

################################################################################

def load_data():
    """
        Load the data from the data directory.

        Returns:
            - Xtr : training dataset
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define the different path
    Xtr_path = os.path.join(os.getcwd(), "Data", "Xtr.csv")
    Xte_path = os.path.join(os.getcwd(), "Data", "Xte.csv")
    Ytr_path = os.path.join(os.getcwd(), "Data", "Ytr.csv")

    # load the data
    Xtr = np.genfromtxt(Xtr_path, delimiter=',')
    Xte = np.genfromtxt(Xte_path, delimiter=',')
    Ytr = np.loadtxt(Ytr_path, delimiter=',', skiprows=1)

    # trim the useless data
    Xtr = Xtr[:,0:3072]
    Xte = Xte[:,0:3072]
    Ytr = Ytr[:,1].astype(np.double)
    logging.info("%d images loaded from file"%Xtr.shape[0])

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

if __name__ == "__main__":
    # load the data
    Xtr, Ytr, Xte = load_data()

    for y in range(0,10):
        id_y = np.where(Ytr==y)[0]
        for j in range(0,10):
            # build image
            im = build_image(Xtr[id_y[j],:])
            im = im - im.min()
            im = im / im.max()

            # display image
            print("label %d - picture %d"%(y,id_y[j]))
            plt.title("label %d - picture %d"%(y,id_y[j]))
            plt.imshow(im, interpolation='nearest')
            plt.show()

    pdb.set_trace()


