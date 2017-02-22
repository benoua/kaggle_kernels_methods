# -*- coding: UTF-8 -*-
# import the necessary packages
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import logging
import cvxopt
from sklearn import svm

################################################################################

def load_data():
    """
        Load the data from the data directory.

        Returns:
            - Xtr : training dataset
            - Xte : testing dataset
            - Ytr : training labels
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define the different path
    Xtr_path = os.path.join(os.getcwd(), "Data", "Xtr.csv")
    Xte_path = os.path.join(os.getcwd(), "Data", "Xte.csv")
    Ytr_path = os.path.join(os.getcwd(), "Data", "Ytr.csv")

    # load the data
    Xtr = np.genfromtxt(Xtr_path, delimiter=',')
    # Xte = np.genfromtxt(Xte_path, delimiter=',')
    Ytr = np.loadtxt(Ytr_path, delimiter=',', skiprows=1)

    # trim the useless data
    Xtr = Xtr[:,0:3072]
    # Xte = Xte[:,0:3072]
    Ytr = Ytr[:,1].astype('uint8')
    logging.info("%d images loaded from file"%Xtr.shape[0])

    return Xtr, Ytr

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

    return hist

################################################################################

def patch_Gram(X, dic, patch_width):
    """
        Generate the Gram matrix corresponding to the patch feature.

        Arguments:
            - X : dataset
            - dic : dictionnary of all words

        Returns:
            - K : Gram matrix
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define path
    gram_path = "gram_patch.csv"

    # check if it already exists
    if os.path.isfile(gram_path):
        # load the matrix
        K = np.loadtxt(gram_path, delimiter=',')
        logging.info("Gram matrix loaded from file")
    else:
        # initialize variables
        hists = np.zeros((X.shape[0], dic.shape[0]))
        K = np.zeros((X.shape[0], X.shape[0]))
        logging.info("Building Gram matrix...")

        # build all histograms
        logging.info("Histograms 0%%")
        for i in range(0, X.shape[0]):
            # retrieve the first histogram
            im_i = build_image(X[i,:])
            hists[i,:] = patch_hist(im_i, dic, patch_width)

            # display progress
            if (i + 1) % (X.shape[0] / 5) == 0:
                p = (i + 1) / (X.shape[0] / 5) * 20
                logging.info("Histograms %d%%"%p)

        # compute the Gram matrix
        K = hists.dot(hists.T)

        # save the matrix
        np.savetxt(gram_path, K, delimiter=',')
        logging.info("Gram matrix saved in file %s"%gram_path)

    return K

################################################################################

def SVM_kernel(K, Y, C):
    """
        Solve the quadratic problem with dual formulation using the kernel Gram
        matrix.

        Quadratic problem for the cvxopt solver:
            min         1/2 x^T P x + q^T x
            s.t.        Gx <= h
                        Ax = b

        Arguments:
            - K : kernel Gram matrix
            - Y : data labels
            - C : regularisation parameter for the SVM soft margin

        Returns:
            - alpha : lagrangian multipliers
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define path
    alpha_path = "alpha_SVM.csv"

    # check if it already exists
    if os.path.isfile(alpha_path):
        # load the data
        alpha = np.loadtxt(alpha_path, delimiter=',')
        logging.info("SVM dual coefs loaded from file")
    else:
        # number of samples
        n = K.shape[0]

        # QP objective function parameters
        P = K * np.outer(Y, Y)
        q = - np.ones(n)

        # QP inequality constraint parameters
        G = np.zeros((2*n, n))
        G[0:n,:] = - np.eye(n)
        G[n:2*n,:] = np.eye(n)
        h = np.zeros((2*n,1))
        h[0:n,0] = 0
        h[n:2*n,0] = C

        # QP equality constraint parameters
        A = Y.reshape(1,n)
        b = np.array([0])

        # convert all matrix to cvxopt matrix
        P_qp = cvxopt.matrix(P.astype(np.double))
        q_qp = cvxopt.matrix(q.astype(np.double))
        G_qp = cvxopt.matrix(G.astype(np.double))
        h_qp = cvxopt.matrix(h.astype(np.double))
        A_qp = cvxopt.matrix(A.astype(np.double))
        b_qp = cvxopt.matrix(b.astype(np.double))

        # solve
        solution = cvxopt.solvers.qp(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp)

        # retrieve lagrangian multipliers
        alpha = Y*np.array(solution['x']).flatten()

        # save the matrix
        np.savetxt(alpha_path, alpha, delimiter=',')
        logging.info("SVM dual coefs saved in file %s"%alpha_path)

    return alpha

################################################################################

def predictor_SVM(alpha, Xtr, dic, patch_width):
    """
        Compute the SVM predictor from the lagrangian multipliers and the
        corresponding histograms.

        Arguments:
            - alpha : lagrangian multipliers
            - Xtr : training dataset
            - dic : dictionnary of all words
            - patch_width : width of the patch

        Returns:
            - predictor : SVM predictor
    """
    # build all histograms
    hists = np.zeros((Xtr.shape[0], dic.shape[0]))
    for i in range(0, Xtr.shape[0]):
        # retrieve the first histogram
        im_i = build_image(Xtr[i,:])
        hists[i,:] = patch_hist(im_i, dic, patch_width)

    # compute predictor
    predictor = alpha.dot(hists)

    # compute prediction score
    prediction = hists.dot(predictor)

    return prediction

################################################################################

def solve_WKRR(K, Y, W, lambd):
    """
        Solve the weighted kernel ridge regression.

        Arguments:
            - K : kernel Gram matrix
            - Y : data labels
            - W : weights
            - lambd : regularization parameter

        Returns:
            - alpha : minimizer
    """
    # retrieve dimensions
    n = K.shape[0]

    # internmediary matrices
    W_sqrt = np.diag(np.sqrt(W.flatten()))
    A = W_sqrt.dot(K).dot(W_sqrt) + lambd * n * np.eye(n)
    A = A.dot(np.diag(np.reciprocal(np.sqrt(W.flatten()))))
    b = W_sqrt.dot(Y.reshape(n,1))

    # compute solution
    alpha = np.linalg.solve(A, b)

    return alpha.flatten()

################################################################################

def logit(u):
    """
        Compute the logit function of a numpy array

        Arguments:
            - u : numpy array

        Returns:
            - res : logit of u
    """
    # compute
    res = np.reciprocal(1 + np.exp(-u))

    return res

################################################################################

def log_reg_kernel(K, Y, lambd):
    """
        Kernel logistic regression

        Arguments:
            - K : kernel Gram matrix
            - Y : data labels
            - lambd : regularization parameter

        Returns:
            - alpha : minimizer
    """
    # initialize variables
    n = K.shape[0]
    alpha = np.zeros(n)
    alpha_prev = np.copy(alpha)
    diff = 1

    # solve with Newton method
    while diff > 1e-6:
        # compute the new solution
        m = K.dot(alpha)
        P = -logit(-Y * m)
        W = logit(m) * logit(-m)
        Z = m + Y * np.reciprocal(logit(-Y * m))
        alpha = solve_WKRR(K, Z, W, lambd)

        # compute the difference
        diff = np.abs(alpha-alpha_prev).sum()
        alpha_prev = np.copy(alpha)

    return alpha.flatten()

################################################################################

if __name__ == "__main__":
    # # load the data
    # Xtr, Ytr = load_data()

    # # build the dictionnary
    # patch_width = 4
    # n_voc = 1000
    # dic = patch_dictionnary(Xtr, n_voc, patch_width)

    # # generate the Gram matrix
    # K = patch_Gram(Xtr, dic, patch_width)

    # Y = np.copy(Ytr).astype(np.double)
    # Y[Y!=3] = -1
    # Y[Y==3] = 1

    # # train the SVM
    # C = 100
    # alpha = SVM_kernel(K, Y, C)
    # prediction = predictor_SVM(alpha, Xtr, dic, patch_width)

    n = 10
    X = np.random.rand(n)*2-1
    Y = np.sign(X)
    X = X + 2
    K = np.outer(X,X)

    alpha = log_reg_kernel(K, Y, 1)

    print("i\ty\ty'\tf(x_i)")
    for i in range(0,n):
        fx = np.sum(alpha * (K[:,i].flatten()))
        print("%d\t%d\t%d\t%f"%(i,Y[i],np.sign(fx),fx))



