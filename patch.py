# -*- coding: UTF-8 -*-
# import the necessary packages
import logging
import numpy as np
from matplotlib import pyplot as plt
from utils import build_image
from utils import raw2gray
from clustering import kmeans
from clustering import gmm
from clustering import compute_gamma

################################################################################

def patch_features(im, width):
    """
        Compute the patch feature representation where we simply extract all the
        4x4 patchs and flatten them resulting in a 64x16 feature descriptor.

        Arguments:
            - im : raw 2D image
            - width : width of the patch

        Returns:
            - feat : feature descriptor of the image
    """
    # define variables
    w = width           # width of the patch
    n = 32/w            # number of grid regions on each side

    # convert to grayscale
    im_gray = raw2gray(im)

    # generate the features
    k = 0
    feat = np.zeros((n**2, w**2))
    for i in range(0, n):
        for j in range(0, n):
            feat[k,:] = im_gray[i*w:(i+1)*w, j*w:(j+1)*w].flatten()
            k += 1

    return feat

################################################################################

def patch_list(X, patch_width):
    """
        List all patch words in the input X.

        Arguments:
            - X : dataset
            - patch_width : width of the patch

        Returns:
            - words : list of all patches of the dataset
    """
    # initialize the list of words
    n_patch = 1024 / patch_width**2
    n_words = X.shape[0] * n_patch
    words = np.zeros((n_words, patch_width**2))

    # go through all image
    for i in range(0, X.shape[0]):
        # extract features
        im = build_image(X[i,:])
        feat = patch_features(im, patch_width)

        # add them to the list
        words[i*n_patch:(i+1)*n_patch,:] = feat

    return words

################################################################################

def patch_dictionnary(X, n_voc, patch_width, show_distrib=False):
    """
        Build the patch dictionnary of the input X containing n_words using
        K-mean clustering.

        Arguments:
            - X : dataset
            - n_voc : number of words contained in the dictionnary
            - patch_width : width of the patch

        Returns:
            - dic : dictionnary of all words
    """
    # build the dictionnary
    from clustering import kmeans
    words = patch_list(X, patch_width)              # compute the words list
    dic, labels = kmeans(words, n_voc, 3, False)    # compute the dictionnary

    # display histogram
    if show_distrib:
        # compute distributiom
        prob, _, _ = plt.hist(labels, bins=n_voc, normed=True)
        sort_id = prob.argsort()[::-1]
        temp = np.zeros(labels.shape)
        uniques = np.unique(labels)
        for i in range(0, uniques.shape[0]):
            temp[labels==sort_id[i]] = i
        cumsum = prob[sort_id].cumsum() / prob.sum()

        # plot the results
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("word number")
        ax1.set_title("patch dictionnary: %d words for %d images"%
            (n_voc,X.shape[0]))
        ax1.hist(temp, bins=n_voc, normed=True, color='b')
        ax1.set_ylabel('individual probability', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(cumsum, color='r')
        ax2.set_ylabel('cumulative probability', color='r')
        ax2.tick_params('y', colors='r')
        plt.savefig("fig/dic_patch_%d_%d.jpg"%(n_voc,X.shape[0]))
        fig.show()

    return dic

################################################################################

def patch_hist(im, dic, patch_width):
    """
        Build the histogram representation of the image.

        Arguments:
            - im : raw 2D image
            - dic : dictionnary of all words
            - patch_width : width of the patch

        Returns:
            - hist : patch histogram
    """
    # retrieve the features
    feat = patch_features(im, patch_width)

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

def patch_hists(X, dic, patch_width):
    """
        Compute histograms for all images.

        Arguments:
            - X : dataset
            - dic : dictionnary of all words
            - patch_width : width of the patch

        Returns:
            - hists : list of histograms of all images
    """
    # build all histograms
    hists = np.zeros((X.shape[0], dic.shape[0]))
    for i in range(0, X.shape[0]):
        # retrieve the first histogram
        im_i = build_image(X[i,:])
        hists[i,:] = patch_hist(im_i, dic, patch_width)

    return hists

################################################################################

def patch_gmm(X, n_mixt, patch_width):
    """
        Build the patch dictionnary of the input X containing n_words using
        K-mean clustering.

        Arguments:
            - X : dataset
            - n_mixt : number of gaussian mixtures
            - patch_width : width of the patch

        Returns:
            - w : weights of each mixture components
            - mu : means of each mixture components
            - sig : covariance matricex of each components
    """
    # build the dictionnary
    words = patch_list(X, patch_width)              # compute the words list
    # w, mu, sig = gaussian_mixture(words, n_mixt)    # compute the gmm
    w, mu, sig = gmm(words, n_mixt)    # compute the gmm

    return w, mu, sig

################################################################################

def patch_fisher(X, w, mu, sig, patch_width):
    """
        Compute histograms for all images.

        Arguments:
            - X : dataset
            - w : weights of each mixture components
            - mu : means of each mixture components
            - sig : covariance matricex of each components
            - patch_width : width of the patch

        Returns:
            - fisher : fisher vectors of all images
    """
    # retrieve dimensions
    n = X.shape[0]
    k = w.shape[0]
    p = patch_width**2

    fisher = np.zeros((n, 2 * k * p))
    # temp = np.zeros((n,k))
    for i in range(0, n):
        # retrieve the first histogram
        im = build_image(X[i,:])
        feat = patch_features(im, patch_width)
        gamma = compute_gamma(feat, w, mu, sig).sum(axis=0)/n
        # temp[i,:] = np.sqrt(gamma / gamma.sum())
        fisher[i,:] = compute_statistics_single(feat, w, mu, sig)
    return fisher
    # return temp
