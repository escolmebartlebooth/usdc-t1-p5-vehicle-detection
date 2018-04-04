"""
    set of functions for feature extraction
"""

# imports
import numpy as np
import cv2
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    """
        function to return HOG features
        Args:
            img: the image to extract features from
            orient, pix_per_cell, cell_per_block: parameters
            vis: whether to return an image with the features visualised
            feature_vec: whether to return a flattened feature set
        Returns:
            features: the features extracted
            hog_image: the visualisation if requested
    """

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def color_hist(img, nbins=32, bins_range=(0, 1)):
    """
        function to return flattened histogram of color channels
        Args:
            img: the image to process
            nbins, bins_range: parameters
        Returns
            hist_features: the flattened channel histogram
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def bin_spatial(img, size=(32, 32)):
    """
        function to return flattened array of image
        Args:
            img: image to process
            size: size to resize image to
        Returns:
            flattened feature set of resized image
    """

    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    return np.hstack((color1, color2, color3))
