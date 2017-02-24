# -*- coding: UTF-8 -*-
# import the necessary packages
import numpy as np

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
    # im_gray = raw2gray(im)
    im_gray = im

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
    magn_top = magn * (bin_top * qudrnt - angle) / qudrnt
    magn_bot = magn * (angle - bin_bot * qudrnt) / qudrnt

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
            hist = hist/np.sqrt(np.square(hist).sum())
            feat[l,:] = hist[:]

            # update iterator
            l += 1

    return feat

################################################################################

if __name__ == "__main__":
    im = np.random.rand(32,32)
    n_bins = 9
    print(HOG_features(im, n_bins).shape)
