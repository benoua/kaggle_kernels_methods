# -*- coding: UTF-8 -*-
# import the necessary packages
import logging
import numpy as np
import pdb
from clustering import *
from patch import *
from svm import *
from utils import *
from HOG import *

import shelve

################################################################################

def cross_validation(feats, kernels, C_array, n, display = True):
    """
        Use cross validation to find C.

        Arguments:
            - feats : training features
            - kernels : array representing kernels where each element is
            composed 'kernel', 'degree' and 'gamma'
            - C_array: values of C to try
            - n : number of tries for each C

        Returns:
            - best_kernel : kernel that minimizes the error
            - best_C : C parameter that minimizes the error
            - best_err : corresponding error
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # intialize parameters
    best_kernel = None
    best_err = 0
    if display: logging.info("kernel\tdeg\tgamma\tC\terror")

    # loop through all kernels
    for kernel in kernels:
        # generate Gram matrix
        K = Gram(feats, feats, kernel=kernel[0], degree=kernel[1],
            gamma=kernel[2])

        # loop through all C
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


                # train our svm
                predictors, masks = SVM_ovo_predictors(K_CV_tr, Y_CV_tr, C)
                K_CV_te = K[perm_te,:]
                K_CV_te = K_CV_te[:,perm_tr]
                Y_CV_pred = SVM_ovo_predict(K_CV_te, predictors, masks)
                

                err += error(Y_CV_pred, Y_CV_te) / n

            # update best parameters
            if display:
                logging.info("%s\t%d\t%0.1f\t%0.2f\t%0.1f"%(kernel[0],kernel[1],
                    kernel[2],C,err))
            if best_kernel is None or err < best_err:
                best_kernel = kernel
                best_C = C
                best_err = err

    if True:
        logging.info("Best parameters:\n\t- kernel: %s\n\t"%best_kernel[0] + \
            "- deg: %d\n\t- gamma: %0.1f"%(best_kernel[1], best_kernel[2]) + \
            "\n\t- C: %0.2f\n\t -error: %0.1f"%(best_C, best_err))

    return best_kernel, best_C, best_err

################################################################################

if __name__ == "__main__":
    # initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # load the data
    Xtr, Ytr, Xte = load_data(5000)
    logging.info("Pictures loaded")
 
    # build the patch hist
    n_voc = 50
    logging.info("n_voc \t %d" %n_voc)
    patch_width = 2
    dic_patch = patch_dictionnary(Xtr, n_voc, patch_width)
    hists_patch = patch_hists(Xtr, dic_patch, patch_width)
    logging.info("Train patch histograms generated")

    # build the HOG hist
    n_voc = 50
    n_bins = 9
    dic_HOG = HOG_dictionnary(Xtr, n_voc, n_bins)
    hists_HOG = HOG_hists(Xtr, dic_HOG, n_bins)
    logging.info("Train HOG histograms generated")

    # build the patch GMM
    n_mixt = 50
    w, mu, sig = patch_gmm(Xtr, n_mixt, patch_width)
    fisher_patch = patch_fisher(Xtr, w, mu, sig, patch_width)
    logging.info("Train patch gmm generated")

    # combine features
    feats_tr = np.hstack((hists_patch, hists_HOG))
    # feats_tr = np.hstack((hists_patch, hists_HOG, fisher_patch))

    # cross validation
    n = 1
    C_array = 0.01*1.5**np.arange(0,24)[::2]

    kernels = [
        ['lin', 0,  0],
        ['pol', 1,  0],
        ['pol', 2,  0],
        ['pol', 3,  0],
        ['pol', 4,  0],
        ['rbf', 0, 0.5],
        ['rbf', 0, 0.7],
        ['rbf', 0, 0.1],
        ['rbf', 0, 1.5],
        ['rbf', 0, 2.0],
        ['rbf', 0, 3.0],
        ['rbf', 0, 3.5],
        ['rbf', 0, 5.0],
        ['rbf', 0, 10.]
        ]

    # Activate ligne below for gridsearch on svm and kernel parameters  
    # kernel, C, err = cross_validation(feats_tr, kernels, C_array, n, True)
    
    # # Best parameters
    kernel = ['rbf', 0, 0.5]
    C = 1.30

    # final prediction
    K_tr = Gram(feats_tr, feats_tr, kernel=kernel[0], degree=kernel[1],
        gamma=kernel[2])
    hists_patch_te = patch_hists(Xte, dic_patch, patch_width)
    hists_HOG_te = HOG_hists(Xte, dic_HOG, n_bins)
    # fisher_patch_te = patch_fisher(Xte, w, mu, sig, patch_width)
    feats_te = np.hstack((hists_patch_te, hists_HOG_te))
    K_te = Gram(feats_te, feats_tr, kernel=kernel[0], degree=kernel[1],
        gamma=kernel[2])
    logging.info("Test features generated")


    # # Test with our predictor with ovo
    predictors, masks = SVM_ovo_predictors(K_tr, Ytr, C)
    Yte = SVM_ovo_predict(K_te, predictors, masks) 

    # Test with our predictor with ova
    # predictors = SVM_ova_predictors(K_tr, Ytr, C)
    # Yte = SVM_ova_predict(K_te, predictors)

    save_submit(Yte)
    logging.info("Done")

