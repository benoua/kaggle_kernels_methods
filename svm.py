# -*- coding: UTF-8 -*-
# import the necessary packages
import logging
import cvxopt
import numpy as np
from scipy.spatial import distance as dist
import pdb

################################################################################

def Gram(feats1, feats2, kernel = 'lin', degree = 3, gamma = None):
    """
        Generate the Gram matrix corresponding to the kernel of the two vectors
        feats1 and feats2.

        Arguments:
            - feats1 : features 1
            - feats2 : features 2
            - kernel : type of kernel used (linear, poly or rbf)
            - degree : degree for polynomial kernel
            - gamma: parameter for rbf kernel

        Returns:
            - K : Gram matrix
    """
    # compute the Gram matrix
    if kernel == 'lin':
        K = feats1.dot(feats2.T)
    elif kernel == 'pol':
        K = (1 + feats1.dot(feats2.T)) ** int(degree)
    elif kernel == 'rbf':
        K = np.exp(-gamma * dist.cdist(feats1, feats2)**2)

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
    cvxopt.solvers.options['reltol'] = 1e-12
    cvxopt.solvers.options['feastol'] = 1e-12
    cvxopt.solvers.options['abstol'] = 1e-12

    # solve
    solution = cvxopt.solvers.qp(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp)

    # retrieve lagrangian multipliers
    alpha = np.array(solution['x']).flatten()

    # compute the intercept
    svp = np.where((alpha<((1-1e-4)*C))*(alpha>(C*1e-4)))[0]
    svn = np.where((alpha>(-(1-1e-4)*C))*(alpha<(-C*1e-4)))[0]
    rhop = 1 - K.dot(alpha)[svp].mean() if svp.size > 0 else np.nan
    rhon = - 1 - K.dot(alpha)[svn].mean() if svn.size > 0 else np.nan
    rho = np.array([rhop, rhon])
    rho = rho[np.isfinite(rho)].mean()

    return alpha, rho

################################################################################

def SVM_ova_predictors(K, Y, C):
    """
        Implement the one versus all strategy for multiclass SVM.

        Arguments:
            - K : kernel Gram matrix
            - Y : data labels
            - C : regularisation parameter for the SVM soft margin

        Returns:
            - predictors : predictor of each SVM
    """
    # retrieve unique labels
    Y_unique = np.unique(Y)
    N = Y_unique.shape[0]

    # go through all labels
    predictors = np.zeros((N, K.shape[0] + 1))
    for i in range(0, Y_unique.shape[0]):
        # select only the required data
        Y_i = np.copy(Y)
        Y_i[Y_i!=Y_unique[i]] = -1
        Y_i[Y_i==Y_unique[i]] = 1

        # compute the corresponding SVM
        logging.info("Training SVM ova for label %d"%Y_unique[i])
        alpha, rho = SVM_kernel(K, Y_i, C)

        # store the values
        predictors[i,0:-1] = alpha
        predictors[i,-1] = rho

    return predictors

################################################################################

def SVM_ova_predict(K, predictors):
    """
        Predict the label using a predictor

        Arguments:
            - K : Gram matrix
            - predictors : predictor of each SVM

        Returns:
            - Ypred : predicted data labels
    """
    # retrieve dimensions
    N = predictors.shape[0]
    n = K.shape[0]

    # retrieve predictors
    alpha = predictors[:,:-1]
    rho = predictors[:,-1]

    # computes scores
    scores = K.dot(alpha.T) + rho
    Ypred = scores.argmax(axis=1)

    return Ypred

#####################################################################

def SVM_ovo_predictors(K, Y, C):
    """
        Implement the one versus one strategy for multiclass SVM.

        Arguments:
            - K : kernel Gram matrix
            - Y : data labels
            - C : regularisation parameter for the SVM soft margin

        Returns:
            - predictors : predictor of each SVM
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


    Y_unique = np.unique(Y)
    N = Y_unique.shape[0]
    # print("nb de unique : %d " %(N))
    # go through all labels
    predictors = np.zeros((N * (N - 1) / 2, K.shape[0] + 1))
    masks = []
    k = 0
    for i in range(0, Y_unique.shape[0]-1):
        for j in range(i+1, Y_unique.shape[0]):
            # build the data mask
            mask = (Y==Y_unique[i]) + (Y==Y_unique[j])
            masks.append(mask)

            # select only the required data
            K_ij = K[mask,:]
            K_ij = K_ij[:,mask]
            Y_ij = Y[mask]
            Y_ij[Y_ij==Y_unique[j]] = -1
            Y_ij[Y_ij==Y_unique[i]] = 1

            # compute the corresponding SVM
            alpha, rho = SVM_kernel(K_ij, Y_ij, C)

            # store the values
            predictors[k,0:-1][mask] = alpha.flatten()
            predictors[k,-1] = rho

            # update iterator
            k = k + 1

    return predictors, masks

################################################################################

def SVM_ovo_predict(K, predictors, masks):
    """
        Predict the label using a predictor

        Arguments:
            - K : Gram matrice of predictors of each SVM size (te, train)
            - predictors 
            - masks used for 1 vs 1 splits

        Returns:
            - Ypred : predicted data labels
    """
    # retrieve dimensions
    N = int( np.sqrt(2 * predictors.shape[0] + 1./4) - 1./2 + 1 )
    # print("check integer : %d "%N)
    n = K.shape[0]
    print(n)

    # loop through all images
    Ypred = np.zeros(n)
    for l in range(0,n):
        # build histogram

        # hist = hists[l]
        K_ = K[l,:]

        # initialize variables
        votes = np.zeros((N,N))
        k = 0

        # loop through all pairs
        for i in range(0,N-1):
            for j in range(i+1,N):
                # retrieve SVM parameters
                mask = masks[k]
                alpha = predictors[k,0:-1][mask].flatten()
                rho = predictors[k,-1]

                # update vote
                # score = w.dot(hist) + rho
                score = K_[mask].dot(alpha.T) + rho
                if score > 0:
                    votes[i,j] = 1
                if score < 0:
                    votes[j,i] = 1

                # update iterator
                k = k + 1
        # classe to win has the most votes
        winners = np.argwhere(votes.sum(axis=1) == np.amax(votes.sum(axis=1))).flatten()
        # choosing randomly the final winner between all winning classes
        Ypred[l] = np.random.choice(winners)

    return Ypred

################################################################################