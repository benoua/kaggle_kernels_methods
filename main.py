# -*- coding: UTF-8 -*-
# import the necessary packages
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import logging
import cvxopt

################################################################################

def load_data():
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
    Xtr = np.genfromtxt(Xtr_path, delimiter=',')
    # Xte = np.genfromtxt(Xte_path, delimiter=',')
    # Ytr = np.loadtxt(Ytr_path, delimiter=',', skiprows=1)

    # trim the useless data
    Xtr = Xtr[:,0:3072]
    # Xte = Xte[:,0:3072]
    # Ytr = Ytr[:,1].astype('uint8')

    return Xtr

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

def kmeans(X, k, n_try):
    """
        Compute the k-mean representation of the input data X.

        Arguments:
            - X : dataset to be clustered
            - k : number of clusters
            - n_try : number of random initialization to perform

        Returns:
            - best_centroids : cluster points
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # do several random initialization and keep the best
    best_centroids = None
    for l in range(0,n_try):
        # initialize variables
        labels = np.zeros(X.shape[0])
        centroids = X[np.random.choice(X.shape[0], k, False), :]
        old_centroids = np.zeros(centroids.shape)
        valid_centroids = True
        label_change = True

        # loop until we reach a local minima
        j = 0
        while not(np.array_equal(old_centroids, centroids)) and label_change:
            # save the old variables before update
            old_centroids = np.copy(centroids)
            old_labels = np.copy(labels)

            # compute new labels with closest centroids
            err_tot = 0
            for i in range(X.shape[0]):
                err = np.square(centroids - X[i,:]).sum(axis=1)
                labels[i] = err.argmin()
                err_tot += err.min()
            err_tot = err_tot

            # display progress
            label_change = np.sum(old_labels != labels)
            logging.info("Kmeans: iter %d-%d, change: %d, error: %0.4f" %
                (l+1, j+1, label_change, err_tot))
            label_change = label_change > 1000
            j += 1

            # check if at least one point is assigned to the centroid
            if not np.array_equal(np.unique(labels), np.arange(0, k)):
                valid_centroids = False
                break

            # recompute the centroids
            for i in range(0, k):
                centroids[i,:] = np.mean(X[labels==i], axis=0)

        # if we have a valid centroid, then compare it to the best
        if valid_centroids:
            if best_centroids is None:
                best_centroids = np.copy(centroids)
                best_err = err_tot
            elif err_tot < best_err:
                best_centroids = np.copy(centroids)
                best_err = err_tot

    logging.info("Kmeans converged.")

    return best_centroids

################################################################################

def patch_dictionnary(X, n_voc, patch_width):
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
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define path
    dic_path = "dic_patch.csv"

    # check if a dictionnary already exists
    if os.path.isfile(dic_path):
        # load the dictionnary
        dic = np.loadtxt(dic_path, delimiter=',')
        logging.info("Dictionnary loaded from file")
    else:
        # build the dictionnary
        words = patch_list(X, patch_width)          # compute the words list
        dic = kmeans(words, n_voc, 1)               # compute the dictionnary
        np.savetxt(dic_path, dic, delimiter=',')    # save the dictionnary
        logging.info("Dictionnary saved in file %s"%dic_path)

    return dic

################################################################################

def patch_hist(im, dic, patch_width):
    """
        Build the histogram representation of the image.

        Arguments:
            - im : raw 2D image
            - dic : dictionnary of all words

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

    return hist

################################################################################

def patch_Graam(X, dic, patch_width):
    """
        Generate the Graam matrix corresponding to the patch feature.

        Arguments:
            - X : dataset
            - dic : dictionnary of all words

        Returns:
            - K : Graam matrix
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define path
    graam_path = "graam_patch.csv"

    # check if it already exists
    if os.path.isfile(graam_path):
        # load the matrix
        K = np.loadtxt(graam_path, delimiter=',')
        logging.info("Graam matrix loaded from file")
    else:
        # initialize variables
        K = np.zeros((X.shape[0], X.shape[0]))

        # loop through all possible pairs
        for i in range(0, X.shape[0]):
            # retrieve the first histogram
            im_i = build_image(X[i,:])
            hist_i = patch_hist(im_i, dic, patch_width)

            # display progress
            logging.info("Graam: iter %d/%d"%(i, X.shape[0]))

            for j in range(i, X.shape[0]):
                # retrieve the second histogram
                im_j = build_image(X[j,:])
                hist_j = patch_hist(im_j, dic, patch_width)

                # compute the Kernel value
                K[i,j] = hist_i.dot(hist_j)
                K[j,i] = hist_i.dot(hist_j)

        # save the matrix
        np.savetxt(graam_path, K, delimiter=',')
        logging.info("Graam matrix saved in file %s"%graam_path)

    return K

################################################################################

def SVM_test(lambda_reg):
    """
        Quadratic problem for the cvxopt solver:
            min         1/2 x^T P x + q^T x
            s.t.        Gx <= h
                        Ax = b

        Quadratic problem for SVM-soft margin:
            min         1/2 x^T K x - y^T x
            s.t.        - y_i x_i <= 0
                        y_i x_i <= 1/(2 lambda n)

        Arguments:
            - lambda_reg : regularisation parameter for the SVM soft margin

        Returns:
            - bla : bla
    """

    # number of samples
    n = 5

    # randomly generate data
    a = np.random.rand(n)
    K = np.outer(a,a)
    Y = 2 * np.random.randint(1, size=n) - 1
    w,v = np.linalg.eig(K)
    print(v)

    # QP objective function parameters
    P = K
    q = -Y

    # QP constraints parameters
    G = np.zeros((2*n, n))
    G[0:n,:] = np.diag(-Y)
    G[n:2*n,:] = np.diag(Y)
    h = np.zeros(2*n)
    h[0:n] = 0
    h[n:2*n] = 1./ (2 * lambda_reg * n)

    # convert all matrix to cvxopt matrix
    P = cvxopt.matrix(P.astype(np.double))
    q = cvxopt.matrix(q.astype(np.double))
    G = cvxopt.matrix(G.astype(np.double))
    h = cvxopt.matrix(h.astype(np.double))

    # solve
    solution = cvxopt.solvers.qp(P, q)
    # solution = cvxopt.solvers.qp(P, q, G, h, A, b)


################################################################################

if __name__ == "__main__":
    # load the data
    # Xtr = load_data()       # testing dataset

    # plt.imshow(raw2gray(im), cmap=plt.get_cmap('gray'))
    # plt.show()

    # # initialize variables
    # patch_width = 4
    # n_voc = 1000

    # # build the dictionnary
    # dic = patch_dictionnary(Xtr, n_voc, patch_width)

    # # generate the Graam matrix
    # K = patch_Graam(Xtr, dic, patch_width)

    # train the SVM
    lambda_reg = 1
    SVM_test(lambda_reg)



