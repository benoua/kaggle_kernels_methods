# -*- coding: UTF-8 -*-
# import the necessary packages
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

################################################################################

def load_data():
    """
        Load the data from the data directory.
    """
    # define the different path
    Xtr_path = os.path.join(os.getcwd(), "Data", "Xtr.csv")
    Xte_path = os.path.join(os.getcwd(), "Data", "Xte.csv")
    Ytr_path = os.path.join(os.getcwd(), "Data", "Ytr.csv")

    # load the data
    Xtr = np.genfromtxt(Xtr_path, delimiter=',')
    Ytr = np.loadtxt(Ytr_path, delimiter=',', skiprows=1)

    # trim the useless data
    Xtr = Xtr[:,0:3072]
    Ytr = Ytr[:,1].astype('uint8')

    return Xtr, Ytr

################################################################################

def build_image(im_line):
    """
        Convert an image from inline representation to matrix BGR.
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

def hist_equalization(im):
    """
        Equalize the image histogram for easier visualization.
    """
    # compute the histogram
    hist, bins = np.histogram(im.flatten(), 256)

    # cumulative distribution function
    cdf = hist.cumsum().astype('float') / hist.sum()

    # interpolate image
    im_eq = np.interp(im.flatten(), bins[:-1], cdf).reshape((32,32,3))

    return im_eq

################################################################################

if __name__ == "__main__":
    # load the data
    Xtr, Ytr = load_data()

    # display a specific label
    label = 1
    lab_idx = np.where(Ytr==label)[0]
    for i in range(0,20):
        im = build_image(Xtr[lab_idx[i],:])
        im_eq = hist_equalization(im)
        plt.imshow(im_eq)
        plt.title("image %d, label %d"%(lab_idx[i],Ytr[lab_idx[i]]))
        plt.show()

    # # visualize it
    # cv2.imshow('image',im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


