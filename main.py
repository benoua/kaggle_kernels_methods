# -*- coding: UTF-8 -*-
# import the necessary packages
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import logging
import cvxopt
from sklearn import svm
from svm_project_utils import plot_dataset, datasets
from scipy import linalg
import pdb


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
        K = np.zeros((X.shape[0], X.shape[0]))

        # loop through all possible pairs
        for i in range(0, X.shape[0]):
            # retrieve the first histogram
            im_i = build_image(X[i,:])
            hist_i = patch_hist(im_i, dic, patch_width)

            # display progress
            logging.info("Gram: iter %d/%d"%(i, X.shape[0]))

            for j in range(i, X.shape[0]):
                # retrieve the second histogram
                im_j = build_image(X[j,:])
                hist_j = patch_hist(im_j, dic, patch_width)

                # compute the Kernel value
                K[i,j] = hist_i.dot(hist_j)
                K[j,i] = hist_i.dot(hist_j)

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
    logging.info("Solving SVM dual formulation")
    solution = cvxopt.solvers.qp(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp)

    # retrieve lagrangian multipliers
    alpha = np.array(solution['x']).flatten()

    return alpha

################################################################################


class quadra_pb(object):
	# Quadratic problem for the cvxopt solver:
	#             min         1/2 x^T P x -  x^T 1n
	#             s.t.        Gx <= h

	#         Arguments:
	#             - K : kernel Gram matrix
	#             - Y : data labels
	#             - C : regularisation parameter for the SVM soft margin

	#         Attributes:
	#             - grad : gradient of objectiv function   
	#             - f_ : objectiv function   
	#             - g_ : indicator function on [0,C]   
	#             - prox_g : proximal opérator of G 
	# 			  - lipschitz_constant : Lipschitz constant of objectives function 

    def __init__(self, K, Y, C):
        self.H = Y * K * np.transpose(Y)
        self.n, self.d = self.H.shape
        self.C = C

    def f_(self, x):
        '''compute f(x)'''
        return 1. / 2 * np.dot( x.T, np.dot (self.H, x) ) - x.sum()    

    def grad(self, x):
        '''compute the gradient of f'''
        return np.dot( self.H , x) - np.ones(self.n)
 
    def grad_i(self, x, i):
        '''coordinate gradient'''
        return (np.dot(self.H[i].T , x) -1 )
    
    def lipschitz_constant(self):
        """Return the Lipschitz constant of the gradient"""
        L = linalg.norm( self.H )
        return L
    
    def lipschitz_constant_i(self):
        """Return the Lipschitz constant for all coordinates"""
        L = [linalg.norm( self.H[i] ) * self.C for i in range(self.n)]
        return L
    
    def prox_g(self, x, s=1, t=1.):
        """Proximal operator for g at x : """
        return x * (x <= self.C) *  (x >= 0) 

    def g_(self, x, s):
        """value of g function, being the indicator function on [0,C] """
        if (x.any() > self.C) + (x.any() < 0) :
            aux = np.inf
        else :
            aux = 0
        return aux

################################################################################



def fista_svm(x0, f, grad_f, g, prox_g, step, n_iter=50, verbose=True, callback = None):
    """FISTA algorithm"""

    x = x0.copy()
    y = x0.copy()
    t = 1.

    for _ in range(n_iter):
                
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        x_new = prox_g( x = y - step * grad_f ( y ), s= step, t=1.)
        y_new = x_new + (t - 1) / t_new * (x_new - x)
        
        t = t_new
        x = x_new
        y = y_new

        # Update metrics after each iteration.
        if callback: 
            callback(x)
    return x



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
    # intialize variables
    predictor = np.zeros(dic.shape[0])

    # loop through all vectors
    for i in range(0, Xtr.shape[0]):
        # retrieve the first histogram
        im = build_image(Xtr[i,:])
        hist = patch_hist(im, dic, patch_width)
        predictor += hist * alpha[i] * kernel(X)  ## modified
 
    print(predictor)
    return predictor

################################################################################


def kernel(X):
	""" 
		Compute the Gram Matrix for any given vector

		Arguments :
			- X : entry data 

		Returns : 
			- Gram Matrix 
	"""
	n = X.shape[0]
	K = np.empty((n, n))
	for i in range(n):
		for j in range(n):
			K[i, j] = np.dot(X[i], X[j])
	return K


def rbf_kernel(X, gamma = None):
	""" 
	Compute the RBF Gram Matrix for any given vector

	Arguments :
	- X : entry data 
	- gamma : float, if None: 1.0 / n_samples_X

	Returns : 
	- Gram Matrix 
	"""
	n = X.shape[0]
	K = np.empty((n, n))
	if gamma is None: 
		gamma = 1. / n

	for i in range(n):
		for j in range(n):
			K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]))  
	return K

##########################################################

def cd(x0, grad_i, prox_g, L, max_iter):
	
	n_features = L.shape[0]
	max_iter = max_iter * n_features

	x = x0.copy()
	for k in range(max_iter + 1):
		#opti of the i-th coordinate
		i = k % n_features

		x[i] -= 1/ L[i] * grad_i(x, i)
		x[i] = prox_g(x[i])

	return x

################################################################################

if __name__ == "__main__":
    # load the data
    # Xtr, Ytr = load_data()       # testing dataset

    # initialize variables
    # patch_width = 4
    # n_voc = 1000

    # build the dictionnary
    # dic = patch_dictionnary(Xtr, n_voc, patch_width)

    # pdb.set_trace()
    # generate the Gram matrix
    # K = patch_Gram(Xtr, dic, patch_width)

    # simulate data
    # n = 10
    # X = np.random.rand(n)
    # Y = 2 * np.random.randint(2, size=n) - 1
    # K = np.outer(X,X)

	# simulate gaussian data BL
	n = 100
	C = 1
	X, Y = datasets(name='gaussian', n_points=n, sigma=1.5)
	X = np.concatenate((X, np.ones((len(X), 1))), axis=1) # adding intercept to X
	K = kernel(X)
	model = quadra_pb(K=K , Y=Y, C=C)


	# Train SVM with intercept
	max_iter = 20000
	# step = 1. / model.lipschitz_constant()
	x_init = np.random.randn(X.shape[0])
	# mu_fista = fista_svm(x_init, model.f_, model.grad, model.g_, model.prox_g,  
	# 	step = step, n_iter=max_iter, verbose=False , callback =None)
	# print(mu_fista)
	L = np.array(model.lipschitz_constant_i())
	x_star = cd(x_init, model.grad_i, model.prox_g, L, max_iter)
	print(x_star[np.abs(x_star) > 1e-5])
	print(np.where(np.abs(x_star)>1e-5))


	# train the SVM
	C = 1
	clf = svm.SVC(kernel='precomputed', C=C)
	clf.fit(K, Y)			
	print(clf.dual_coef_)
	print(np.sort(clf.support_))


	alpha = SVM_kernel(K, Y, C)
	print(alpha[np.abs(alpha) > 1e-5])
	print(np.where(np.abs(alpha)>1e-5))



    # compute the predictor
    # predictor = predictor_SVM(alpha, Xtr, dic, patch_width)


################################################################################

# def one_vs_all_svms (K, Y, C):
# 	""" 
# 		Multilabel classification SVM : train all classifiers with a one vs all rule.

#         Arguments:
#             - K : kernel Gram matrix
#             - Y : data labels
#             - C : regularisation parameter for the SVM soft margin

# 		Returns : 
# 			- Most probable class 
# 	"""
# 	num_classes = np.max(Y) + 1
# 	svm_classifiers = []
# 	alphas = []
# 	for i in range(num_classes):

# 		idxs_i = Y == i
# 		Y[idxs_i] = 1
# 		Y[-idxs_i] = -1 

# 		alphas.append(SVM_kernel(K, Y, C))

# 		# svm_classifiers.append .. add the found classifier in this list. 






	
