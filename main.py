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
        dic = kmeans(words, n_voc, 3)               # compute the dictionnary
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
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define path
    hists_path = "hists_patch.csv"

    # check if it already exists
    if os.path.isfile(hists_path):
        # load the matrix
        hists = np.loadtxt(hists_path, delimiter=',')
        logging.info("Histogram list loaded from file")
    else:
        logging.info("Histograms 0%%")

        # build all histograms
        hists = np.zeros((X.shape[0], dic.shape[0]))
        for i in range(0, X.shape[0]):
            # retrieve the first histogram
            im_i = build_image(X[i,:])
            hists[i,:] = patch_hist(im_i, dic, patch_width)

            # display progress
            if (i + 1) % (X.shape[0] / 5) == 0:
                p = (i + 1) / (X.shape[0] / 5) * 20
                logging.info("Histograms %d%%"%p)

        # save the list
        np.savetxt(hists_path, hists, delimiter=',')
        logging.info("Histogram list saved in file %s"%hists_path)

    return hists

################################################################################

def patch_Gram(X, dic, patch_width, hists):
    """
        Generate the Gram matrix corresponding to the patch feature.

        Arguments:
            - X : dataset
            - dic : dictionnary of all words
            - patch_width : width of the patch

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
        # compute the Gram matrix
        K = hists.dot(hists.T)

        # save the matrix
        np.savetxt(gram_path, K, delimiter=',')
        logging.info("Gram matrix saved in file %s"%gram_path)

    return K

################################################################################

def SVM_kernel(K, Y, C, hists):
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
            - hists : list of histograms of all images

        Returns:
            - w : vector in feature space
            - rho : intercept
    """
    # number of samples
    n = K.shape[0]

    # QP objective function parameters
    P = K
    q = -Y

    # QP inequality constraint parameters
    G = np.zeros((2*n, n))
    G[0:n,:] = - np.diag(Y)
    G[n:2*n,:] = np.diag(Y)
    h = np.zeros((2*n,1))
    h[0:n,0] = 0
    h[n:2*n,0] = C

    # QP equality constraint parameters
    A = np.ones((1,n))
    b = np.array([0])

    # convert all matrix to cvxopt matrix
    P_qp = cvxopt.matrix(P.astype(np.double))
    q_qp = cvxopt.matrix(q.astype(np.double))
    G_qp = cvxopt.matrix(G.astype(np.double))
    h_qp = cvxopt.matrix(h.astype(np.double))
    A_qp = cvxopt.matrix(A.astype(np.double))
    b_qp = cvxopt.matrix(b.astype(np.double))

    # hide outputs
    cvxopt.solvers.options['show_progress'] = False

    # solve
    solution = cvxopt.solvers.qp(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp)

    # retrieve lagrangian multipliers
    alpha = np.array(solution['x']).flatten()

    # compute the intercept
    fx  = K.dot(alpha.reshape(n,1))
    rho = -0.5 * (fx[Y>0].min() + fx[Y<0].max())

    # compute the representation
    w = alpha.reshape(1,n).dot(hists).flatten()

    return w, rho

################################################################################

def SVM_predictors(K, Y, C, hists):
    """
        Implement the one versus one strategy for multiclass SVM.

        Arguments:
            - K : kernel Gram matrix
            - Y : data labels
            - C : regularisation parameter for the SVM soft margin
            - hists : list of histograms of all images

        Returns:
            - predictors : predictor of each SVM
    """
    # retrieve unique labels
    Y_unique = np.unique(Y)
    N = Y_unique.shape[0]

    # go through all labels
    predictors = np.zeros((N * (N - 1) / 2, hists.shape[1] + 1))
    k = 0
    for i in range(0, Y_unique.shape[0]-1):
        for j in range(i+1, Y_unique.shape[0]):
            # build the data mask
            mask = (Y==Y_unique[i]) + (Y==Y_unique[j])

            # select only the required data
            K_ij = K[mask,:]
            K_ij = K_ij[:,mask]
            hists_ij = hists[mask,:]
            Y_ij = Y[mask]
            Y_ij[Y_ij==Y_unique[j]] = -1
            Y_ij[Y_ij==Y_unique[i]] = 1

            # compute the corresponding SVM
            w, rho = SVM_kernel(K_ij, Y_ij, C, hists_ij)

            # store the values
            predictors[k,0:-1] = w.flatten()
            predictors[k,-1] = rho

            # update iterator
            k = k + 1

    return predictors

################################################################################

def SVM_predict(X, predictors, dic, patch_width):
    """
        Predict the label using a predictor

        Arguments:
            - X : dataset
            - predictors : predictor of each SVM
            - dic : dictionnary of all words
            - width : width of the patch

        Returns:
            - Ypred : predicted data labels
    """
    # retrieve dimensions
    N = int(np.sqrt(2 * predictors.shape[0] + 1./4) - 1./2)
    n = X.shape[0]

    # loop through all images
    Ypred = np.zeros(n)
    for l in range(0,n):
        # build histogram
        im = build_image(X[l,:])
        hist = patch_hist(im, dic, patch_width)

        # initialize variables
        votes = np.zeros((N,N))
        k = 0

        # loop through all pairs
        for i in range(0,N-1):
            for j in range(i+1,N):
                # retrieve SVM parameters
                w = predictors[k,0:-1].flatten()
                rho = predictors[k,-1]

                # update vote
                score = w.dot(hist) + rho
                votes[i,j] = score
                votes[j,i] = -score

                # update iterator
                k = k + 1

        Ypred[l] = np.sign(votes).sum(axis=1).argmax()

    return Ypred

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

if __name__ == "__main__":
    # load the data
    Xtr, Ytr, Xte = load_data()

    # build the dictionnary
    patch_width = 4
    n_voc = 1000
    dic = patch_dictionnary(Xtr, n_voc, patch_width)

    # generate histograms list
    hists = patch_hists(Xtr, dic, patch_width)

    # generate the Gram matrix
    K = patch_Gram(Xtr, dic, patch_width, hists)

    # # find best C
    # print("C\tError\tsk-learn")
    # for C in [0.01, 0.05, 0.1, 0.5, 1, 10, 100, 500, 1000]:
    #     # train our SVM
    #     predictors = SVM_predictors(K, Ytr, C, hists)
    #     Ypred = SVM_predict(Xtr, predictors, dic, patch_width)

    #     # train their SVM
    #     clf = svm.SVC(kernel='precomputed', C=C)
    #     clf.fit(K, Ytr)

    #     # display results
    #     print("%0.2f\t%d\t%d"%(C, error(Ypred, Ytr), error(clf.predict(K),Ytr)))

    C = 1
    predictors = SVM_predictors(K, Ytr, C, hists)
    Ypred = SVM_predict(Xte, predictors, dic, patch_width)
    n = Ypred.shape[0]
    data = np.hstack((np.arange(1,n+1).reshape(n,1),Ypred.reshape(n,1)))
    np.savetxt("test.csv", data, header="Id,Prediction", comments="", fmt="%d",
        delimiter=',')

