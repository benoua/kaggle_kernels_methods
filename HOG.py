# -*- coding: UTF-8 -*-
# import the necessary packages
import numpy as np
from matplotlib import pyplot as plt
from utils import build_image
from utils import raw2gray
from clustering import kmeans

################################################################################

def HOG_features(im, n_bins):
    """
        Compute the HOG feature representation.

        Arguments:
            - im : raw 2D image
            - n_bins : number of angle bins

        Returns:
            - feat : feature descriptor of the image
    """
    # initialize variables
    qudrnt = 360. / n_bins
    n_block = (im.shape[0] / 4 - 1) * (im.shape[1] / 4 - 1)
    feat = np.zeros((n_block, n_bins))

    # convert RGB 2 gray image
    im_gray = raw2gray(im)

    # compute gradient and store it as complex
    grad = np.zeros(im_gray.shape).astype(np.complex)
    grad[:,1:-1] += 0.5 * (im_gray[:,2:] - im_gray[:,:-2])
    grad[:,0]    += im_gray[:,1] - im_gray[:,0]
    grad[:,-1]   += im_gray[:,-1] - im_gray[:,-2]
    grad[1:-1,:] += 0.5 * 1j * (im_gray[2:,:] - im_gray[:-2,:])
    grad[0,:]    += 1j * (im_gray[1,:] - im_gray[0,:])
    grad[-1,:]   += 1j * (im_gray[-1,:] - im_gray[-2,:])

    # retrieve angle and magnitude
    angle = np.angle(grad)*360/2/np.pi + 180
    magn  = np.abs(grad)

    # find corresponding bins
    bin_top = np.ceil(angle / qudrnt)
    bin_bot = np.floor(angle / qudrnt)

    # compute proportial magnitude
    magn_bot = magn * (bin_top * qudrnt - angle) / qudrnt
    magn_top = magn * (angle - bin_bot * qudrnt) / qudrnt

    l = 0
    for i in range(0,im.shape[0] / 4 - 1):
        for j in range(0,im.shape[1] / 4 - 1):
            # flatten vectors
            bin_top_crop  = bin_top[4*i:4*(i+1),4*j:4*(j+1)].flatten()
            bin_bot_crop  = bin_bot[4*i:4*(i+1),4*j:4*(j+1)].flatten()
            magn_top_crop = magn_top[4*i:4*(i+1),4*j:4*(j+1)].flatten()
            magn_bot_crop = magn_bot[4*i:4*(i+1),4*j:4*(j+1)].flatten()

            # compute the histogram
            hist = np.zeros(n_bins)
            for k in range(0,n_bins):
                hist[k] += magn_top_crop[bin_top_crop==k].sum()
                hist[k] += magn_bot_crop[bin_bot_crop==k].sum()

            # normalize histogram
            norm = np.sqrt(np.square(hist).sum())
            hist = hist/norm if norm > 0 else hist
            feat[l,:] = hist[:]

            # update iterator
            l += 1

    return feat

################################################################################

def HOG_list(X, n_bins):
    """
        List all HOG words in the input X.

        Arguments:
            - X : dataset
            - n_bins : number of angle bins

        Returns:
            - words : list of all patches of the dataset
    """
    # initialize the list of words
    n_block = (32 / 4 - 1) * (32 / 4 - 1)
    n_words = X.shape[0] * n_block
    words = np.zeros((n_words, n_bins))

    # go through all image
    for i in range(0, X.shape[0]):
        # extract features
        im = build_image(X[i,:])
        feat = HOG_features(im, n_bins)

        # add them to the list
        words[i*n_block:(i+1)*n_block,:] = feat

    return words

################################################################################

def HOG_dictionnary(X, n_voc, n_bins, show_distrib=False):
    """
        Build the patch dictionnary of the input X containing n_words using
        K-mean clustering.

        Arguments:
            - X : dataset
            - n_voc : number of words contained in the dictionnary
            - n_bins : number of angle bins

        Returns:
            - dic : dictionnary of all words
    """
    # build the dictionnary
    words = HOG_list(X, n_bins)                     # compute the words list
    dic, labels = kmeans(words, n_voc, 3, False)    # compute the dictionnary

    # display histogram
    if show_distrib:
        plt.hist(labels, bins=n_voc, normed=True)
        plt.title("HOG dictionnary %d words for %d images"%(n_voc,X.shape[0]))
        plt.xlabel("word number")
        plt.ylabel("probability")
        plt.savefig("fig/dic_HOG_%d_%d.jpg"%(n_voc,X.shape[0]))
        plt.show()

    return dic

################################################################################

def HOG_hist(im, dic, n_bins):
    """
        Build the histogram representation of the image.

        Arguments:
            - im : raw 2D image
            - dic : dictionnary of all words
            - n_bins : number of angle bins

        Returns:
            - hist : HOG histogram
    """
    # retrieve the features
    feat = HOG_features(im, n_bins)

    # compute the histogram
    hist = np.zeros(dic.shape[0])
    for i in range(0, feat.shape[0]):
        # find the closest dictionnary word
        n_word = np.square(dic - feat[i,:]).sum(axis=1).argmin()
        hist[n_word] += 1

    # normalize histogram
    hist = np.sqrt(hist / hist.sum())
    # hist = hist / hist.sum()

    return hist

################################################################################

def HOG_hists(X, dic, n_bins):
    """
        Compute histograms for all images.

        Arguments:
            - X : dataset
            - dic : dictionnary of all words
            - n_bins : number of angle bins

        Returns:
            - hists : list of histograms of all images
    """
    # build all histograms
    hists = np.zeros((X.shape[0], dic.shape[0]))
    for i in range(0, X.shape[0]):
        # retrieve the first histogram
        im_i = build_image(X[i,:])
        hists[i,:] = HOG_hist(im_i, dic, n_bins)

    return hists
