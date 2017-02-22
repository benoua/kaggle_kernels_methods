import numpy as np
import cvxopt
from sklearn import svm

############### DATA SIMULATION ###############
X = np.array([-5, 1, -9, 2, 0, 4, -0.1])
Y = np.array([-1, 1, 1, 1, 1, 1, -1])
n = X.shape[0]
K = np.outer(X,X)

################# SKLEARN SVM #################
# train the sklearn SVM
C = 100
clf = svm.SVC(kernel='precomputed', C=C)
clf.fit(K, Y)

# retrieve lagrangian multipliers
alpha_sk = np.zeros(n)
alpha_sk[clf.support_] = clf.dual_coef_

################# MANUAL SVM ##################
# QP objective function parameters
P = K * np.outer(Y, Y)
q = - np.ones(n)
G = np.zeros((2*n, n))
G[0:n,:] = - np.eye(n)
G[n:2*n,:] = np.eye(n)
h = np.zeros((2*n,1))
h[0:n,0] = 0
h[n:2*n,0] = C
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
alpha_manual = Y*np.array(solution['x']).flatten()
alpha_manual[np.abs(alpha_manual)<1e-3] = 0

################### RESULTS ###################
print(np.concatenate((alpha_manual.reshape(n,1),alpha_sk.reshape(n,1)),axis = 1))