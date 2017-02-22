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
            - alpha : minimizer in RHKS
            - rho : intercept
    """
    # initialize variables
    n = K.shape[0]
    alpha = np.zeros(n)
    alpha_prev = np.copy(alpha)
    diff = 1

    # solve alpha with Newton method
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

    # compute the intercept rho
    fx = (K * alpha).sum(axis=1)
    rho = -0.5 * (fx[Y>0].min() + fx[Y<0].max())

    return alpha.flatten(), rho