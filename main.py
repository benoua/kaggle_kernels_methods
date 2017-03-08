# -*- coding: UTF-8 -*-
# import the necessary packages
import os
import logging
import cvxopt
import numpy as np
import sys
import pdb
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from sklearn import svm
from sklearn.mixture import GaussianMixture

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
            - labels : corresponding association
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
            label_change = label_change > X.shape[0]*5./100
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

    return best_centroids, labels

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
    words = patch_list(X, patch_width)          # compute the words list
    dic, labels = kmeans(words, n_voc, 3)       # compute the dictionnary

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
        ax1.set_title("patch dictionnary: %d words for %d images"%(n_voc,X.shape[0]))
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

def Gram(hists, kernel = 'linear', degree = 3, gamma = None):
    """
        Generate the Gram matrix corresponding to the patch feature.

        Arguments:
            - hists : histogramme of the image among patches
            - kernel : type of kernel used (linear, poly or rbf)
            - degree : degree for polynomial kernel
            - gamma: parameter for rbf kernel

        Returns:
            - K : Gram matrix
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # define path
    if kernel == 'linear':
        gram_path = "gram_linear.csv"
    elif kernel == 'poly':
        gram_path = "gram_poly_degree_%d.csv" % degree
    elif kernel =='rbf':
        gram_path = "gram_rbf_gamma_%0.3f.csv" % gamma
    else:
        print('kernel name \'%s\' not defined. Script aborted.'%kernel)
        sys.exit(1)

    # compute the Gram matrix
    if kernel == 'linear':
        K = hists.dot(hists.T)
    elif kernel == 'poly':
        K = (1 + hists.dot(hists.T)) ** int(degree)
    elif kernel == 'rbf':
        K = np.exp(-gamma * dist.squareform(dist.pdist(hists, 'euclidean'))**2)

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
    svp = np.where((alpha<(0.99*C))*(alpha>(0.01*C)))[0]
    svn = np.where((alpha>(-0.99*C))*(alpha<(-0.01*C)))[0]
    rhop = 1 - K.dot(alpha)[svp].mean() if svp.size > 0 else np.nan
    rhon = - 1 - K.dot(alpha)[svn].mean() if svn.size > 0 else np.nan
    rho = np.array([rhop, rhon])
    rho = rho[np.isfinite(rho)].mean()

    # compute the representation
    w = alpha.reshape(1,n).dot(hists).flatten()

    return w, rho

################################################################################

def SVM_ovo_predictors(K, Y, C, hists):
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

def SVM_ova_predictors(K, Y, C, hists):
    """
        Implement the one versus all strategy for multiclass SVM.

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
    predictors = np.zeros((N, hists.shape[1] + 1))
    for i in range(0, Y_unique.shape[0]):
        # select only the required data
        Y_i = np.copy(Y)
        Y_i[Y_i!=Y_unique[i]] = -1
        Y_i[Y_i==Y_unique[i]] = 1

        # compute the corresponding SVM
        logging.info("Training SVM ova for label %d"%Y_unique[i])
        w, rho = SVM_kernel(K, Y_i, C, hists)

        # store the values
        predictors[i,0:-1] = w.flatten()
        predictors[i,-1] = rho

    return predictors

################################################################################

def SVM_ovo_predict(hists, predictors):
    """
        Predict the label using a predictor

        Arguments:
            - predictors : predictor of each SVM
            - hists : histogram of the data to predict

        Returns:
            - Ypred : predicted data labels
    """
    # retrieve dimensions
    N = 1. / 2 + np.sqrt(2 * predictors.shape[0] + 1./4)
    n = hists.shape[0]

    # loop through all images
    Ypred = np.zeros(n)
    for l in range(0,n):
        # build histogram
        hist = hists[l]

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
                if score > 0:
                    votes[i,j] = 1
                if score < 0:
                    votes[j,i] = 1

                # update iterator
                k = k + 1
        # classe to win has the most votes
        winners = np.argwhere(votes.sum(axis=1) == np.amax(votes.sum(axis=1)))
        winners = winners.flatten()
        # choosing randomly the final winner between all winning classes
        Ypred[l] = np.random.choice(winners)

    return Ypred

################################################################################

def SVM_ova_predict(hists, predictors):
    """
        Predict the label using a predictor

        Arguments:
            - hists : histogram of the data to predict
            - predictors : predictor of each SVM

        Returns:
            - Ypred : predicted data labels
    """
    # retrieve dimensions
    N = predictors.shape[0]
    n = hists.shape[0]

    # loop through all images
    Ypred = np.zeros(n)
    for l in range(0,n):
        # initialize variables
        hist = hists[l,:]
        votes = np.zeros(N)

        # loop through all pairs
        for i in range(0,N):
            # retrieve SVM parameters
            w = predictors[i,0:-1].flatten()
            rho = predictors[i,-1]

            # update vote
            score = w.dot(hist) + rho
            votes[i] = score

        Ypred[l] = votes.argmax()

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
    words = HOG_list(X, n_bins)                 # compute the words list
    dic, labels = kmeans(words, n_voc, 3)       # compute the dictionnary

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

################################################################################

def gaussian_mixture(X, k=100):
    """
        Gaussian mixture implementation

        Arguments:
            - X : data input stored in row vectors
            - k : number of components for GMM

        Returns:
            - w : weights of each mixture components
            - mu : means of each mixture components
            - sig : covariance matricex of each components
    """
    # compute the GMM using scikit learn
    g = GaussianMixture(n_components=k, covariance_type='diag')
    g.fit(X)
    logging.info("GMM trained")

    # retrieve results
    w = g.weights_
    mu = g.means_
    sig = g.covariances_

    return w, mu, sig

################################################################################

def gmm(x, k, n_init=1):
    """
        Gaussian mixture implementation

        Arguments:
            - X : data input stored in row vectors
            - k : number of components for GMM
            - n_init : number of initializations for kmeans

        Returns:
            - w : weights of each mixture components
            - mu : means of each mixture components
            - sig : covariance matricex of each components

    """

    # Setup the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("---- EM algorithm for identity covariance ----")

    # Declare the variables used for storing the data
    n = x.shape[0]
    p = x.shape[1]

    # Initialize the variable for the EM algirithm using k-means
    logging.info("---- K-mean initialisation ----")
    centroids, labels = kmeans(x, k, n_init)

    q = np.zeros((n,k))
    mu = np.zeros((k,p))
    mu = np.copy(centroids)
    sigma = np.random.rand(k,p)
    pi = np.zeros(k)

    for i in np.arange(0, k):
        pi[i] = np.sum(labels==i)/float(n)
    mu_old = np.zeros((k,p))
    sigma_old = np.zeros((k,p))
    logging.info("---- K-mean initialisation ended ----")

    # Loop
    logging.info("---- EM steps looping ----")
    l = 0;
    while np.sum(np.square(mu_old-mu))+np.sum(np.square(sigma_old-sigma))>1e-6:
        # Record old values
        mu_old = np.copy(mu)
        sigma_old = np.copy(sigma)

        '''E - STEP'''
        for i in np.arange(0,k):

            sig_ = np.diag(sigma[i,:])
            mu_ = mu[i]
            pi_ = pi[i]

            q[:,i] = np.log(pi_) + multivariate_normal.logpdf(x, mu_, sig_)

        q = q - logsumexp(q, axis = 1).reshape(q.shape[0], 1)

        q = np.exp(q)


        '''M - STEP'''
        pi = np.sum(q, axis=0)


        for i in np.arange(0,k):
            mu[i,:] = np.sum(x*q[:,i].reshape((x.shape[0],1)), axis=0)/pi[i]

            # sigma
            for j in range(p) :
                sum_ = 0
                for s in range(n):

                    sum_ += (x[s,j] - mu[i,j])**2 * q[s,i]
                sigma[i,j] = sum_ / pi[i]

        pi = pi/np.sum(pi)

        # Update the number of iterations
        l += 1

    logging.info("EM algorithm converged after %d iterations"%l)

    return pi, mu, sigma


################################################################################

def compute_gamma(X, weights, means, covariances):
    """
        Compute the soft assignment to the GMM for each row vector in X.

        Arguments:
            - weights : weights of each mixture components
            - means : means of each mixture components
            - covariances : covariance matricex of each components

        Returns:
            - gamma : probability of belonging to a GMM

    """
    # intialize variables
    k = weights.shape[0]
    n = X.shape[0]
    gamma = np.zeros((n, k))

    # loop through all mixtures
    for i in range(0,k):
        sig = np.diag(covariances[i])
        mu = means[i]
        weight = weights[i]
        gamma[:,i] = np.log(multivariate_normal.pdf(X,mu,sig)) + np.log(weight)

    # normalize probabilities
    gamma = gamma - logsumexp(gamma, axis=1).reshape(gamma.shape[0],1)
    gamma = np.exp(gamma)

    return gamma

################################################################################

def compute_statistics_single(X, weights, means, covariances):
    """
        Compute the statistics of the generative model for one element.

        Arguments:
            - X : single data input
            - weights : weights of each mixture components
            - means : means of each mixture components
            - covariances : covariance matricex of each components

        Returns:
            - phi : representation in row vector
    """
    # intialize variables
    k = weights.shape[0]
    n = X.shape[0]
    p = X.shape[1]
    phi = np.zeros((k,2 * p))

    # compute soft assignments
    gamma = compute_gamma(X, weights, means, covariances)

    # compute representation for each mixture
    for j in range(0,k):
        mu        = means[j]
        sig       = covariances[j]
        weight    = weights[j]
        gammaj    = gamma[:,j].reshape((n,1))
        phi_mu    = np.sum(gammaj * (X - mu) / sig, axis=0)
        phi_mu   /= (n*np.sqrt(weight))
        phi_sig   = np.sum(gammaj * ((X - mu)**2 / sig**2 - 1), axis=0)
        phi_sig  /= (n*np.sqrt(2 * weight))
        phi[j,:p] = phi_mu
        phi[j,p:] = phi_sig

    # flatten representation
    phi = phi.flatten()

    return phi

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
    w, mu, sig = gaussian_mixture(words, n_mixt)    # compute the gmm

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

    # fisher = np.zeros((n, 2 * k * p))
    temp = np.zeros((n,k))
    for i in range(0, n):
        # retrieve the first histogram
        im = build_image(X[i,:])
        feat = patch_features(im, patch_width)
        gamma = compute_gamma(feat, w, mu, sig).sum(axis=0)/n
        temp[i,:] = np.sqrt(gamma / gamma.sum())
        # fisher[i,:] = compute_statistics_single(feat, w, mu, sig)
    # return fisher
    return temp

################################################################################

def cross_validation(K, n, C_array, display = True):
    """
        Use cross validation to find C.

        Arguments:
            - K : Gram matrix
            - n : number of tries for each C
            - C_array: values of C to try

        Returns:
            - best_C : C parameter that minimizes the error
            - best_err : corresponding error
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # find best C
    best_C = None
    if display: logging.info("C\tError")
    for C in C_array:
        err = 0
        for i in range(0,n):
            # split the data
            perm = np.random.permutation(K.shape[0])
            perm_tr = perm[:int(K.shape[0]*0.8)]
            perm_te = perm[int(K.shape[0]*0.8):]
            Y_CV_tr = Ytr[perm_tr]
            Y_CV_te = Ytr[perm_te]
            hists_CV_tr = feats[perm_tr,:]
            hists_CV_te = feats[perm_te,:]
            K_CV_tr = K[perm_tr,:]
            K_CV_tr = K_CV_tr[:,perm_tr]

            # train scikit SVM
            clf = svm.SVC(kernel='precomputed', C=C)
            clf.fit(K_CV_tr, Y_CV_tr)
            K_CV_te = K[perm_te,:]
            K_CV_te = K_CV_te[:,perm_tr]
            Y_CV_pred = clf.predict(K_CV_te)

            err += error(Y_CV_pred, Y_CV_te)
        err = err / n
        if display: logging.info("%0.3f\t%0.1f"%(C, err))
        if best_C is None or err < best_err:
            best_err = err
            best_C = C
    if display: logging.info("Best C: %0.3f, error: %0.1f"%(best_C, best_err))

    return best_C, best_err

################################################################################

if __name__ == "__main__":
    # load the data
    Xtr, Ytr, Xte = load_data(1000)

    # build the patch hist
    n_voc = 200
    patch_width = 4
    dic_patch = patch_dictionnary(Xtr, n_voc, patch_width, True)
    hists_patch = patch_hists(Xtr, dic_patch, patch_width)

    # # build the HOG hist
    # n_voc = 200
    # n_bins = 9
    # dic_HOG = HOG_dictionnary(Xtr, n_voc, n_bins, True)
    # hists_HOG = HOG_hists(Xtr, dic_HOG, n_bins)

    # # build the patch GMM
    # n_mixt = 200
    # patch_width = 4
    # w_patch, mu_patch, sig_patch = patch_gmm(Xtr, n_mixt, patch_width)
    # fisher_patch = patch_fisher(Xtr, w_patch, mu_patch, sig_patch, patch_width)

    # combine features
    # feats = np.hstack((hists_patch, hists_HOG))
    # feats = hists_patch
    feats = fisher_patch

    # generate the Gram matrix
    # K = feats.dot(feats.T)
    K1 = Gram(feats, kernel = 'linear')
    K2 = Gram(feats, kernel = 'poly', degree = 1)
    K3 = Gram(feats, kernel = 'poly', degree = 2)
    K4 = Gram(feats, kernel = 'poly', degree = 3)
    K5 = Gram(feats, kernel = 'poly', degree = 4)
    K6 = Gram(feats, kernel = 'poly', degree = 5)
    K7 = Gram(feats, kernel = 'rbf', gamma = 0.5)
    K8 = Gram(feats, kernel = 'rbf', gamma = 0.75)
    K9 = Gram(feats, kernel = 'rbf', gamma = 1)
    K10 = Gram(feats, kernel = 'rbf', gamma = 1.5)
    K11 = Gram(feats, kernel = 'rbf', gamma = 2)
    K12 = Gram(feats, kernel = 'rbf', gamma = 2.5)

    # cross validation
    n = 10
    C_array = 0.01*1.5**np.arange(0,30)
    i = 0
    print("K\terror\tC")
    for K in [K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11]:
        best_C, best_err = cross_validation(K, n, C_array, False)
        print("K%d\t%0.1f\t%0.2f"%(i,best_err,best_C))

    # # final prediction
    # C = best_C
    # predictors = SVM_ova_predictors(K, Ytr, C, feats)
    # hists_patch_te = patch_hists(Xte, dic_patch, patch_width)
    # hists_HOG_te = HOG_hists(Xte, dic_HOG, n_bins)
    # hists_te = np.hstack((hists_patch_te, hists_HOG_te))
    # Yte = SVM_ova_predict(hists_te, predictors)
    # save_submit(Yte)


