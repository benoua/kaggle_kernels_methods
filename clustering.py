# -*- coding: UTF-8 -*-
# import the necessary packages
import logging
import numpy as np
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

################################################################################

def kmeans(X, k, n_try, display = True):
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
            if display:
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

    if display: logging.info("Kmeans converged.")

    return best_centroids, labels



################################################################################

def gmm(x, k, n_init=1, tol = 1e-6, max_iter = 100):
    """
        Gaussian mixture implementation

        Arguments:
            - X : data input stored in row vectors
            - k : number of components for GMM
            - n_init : number of initializations for kmeans
            - tol : max precision before stoping algo
            - max_iter : number max of EM iterations

        Returns:
            - w : weights of each mixture components
            - mu : means of each mixture components
            - sig : covariance matricex of each components

    """
    # declare the variables used for storing the data
    n = x.shape[0]
    p = x.shape[1]

    # initialize the variable for the EM algirithm using k-means
    centroids, labels = kmeans(x, k, n_init, False)

    q = np.zeros((n,k))
    mu = np.zeros((k,p))
    mu = np.copy(centroids)
    sigma = np.random.rand(k,p)
    pi = np.zeros(k)

    for i in np.arange(0, k):
        pi[i] = np.sum(labels==i)/float(n)
    mu_old = np.zeros((k,p))
    sigma_old = np.zeros((k,p))

    # loop
    l = 0;
    while np.sum(np.square(mu_old-mu))+np.sum(np.square(sigma_old-sigma))>tol \
     and l < max_iter:
        # record old values
        mu_old = np.copy(mu)
        sigma_old = np.copy(sigma)

        # expectation step
        for i in np.arange(0,k):
            sig_ = np.diag(sigma[i,:])
            mu_ = mu[i]
            pi_ = pi[i]
            q[:,i] = np.log(pi_) + multivariate_normal.logpdf(x, mu_, sig_)

        # normalize q
        q = q - logsumexp(q, axis = 1).reshape(q.shape[0], 1)
        q = np.exp(q)

        # maximization step
        pi = np.sum(q, axis=0)
        for i in np.arange(0,k):
            # mean
            mu[i,:] = np.sum(x*q[:,i].reshape((x.shape[0],1)), axis=0)/pi[i]

            # covariance
            for j in range(p) :
                temp = np.sum( ((x[:,j] - mu[i, j])**2 ) * q[:,i], axis = 0)
                sigma[i,j] = temp / pi[i]

        # weight
        pi = pi/np.sum(pi)

        # Update the number of iterations
        l += 1

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
