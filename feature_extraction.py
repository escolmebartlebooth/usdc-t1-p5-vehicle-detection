"""
    set of functions for feature extraction
"""

# imports
import numpy as np
import cv2
from skimage.feature import hog

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
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

def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
        function to extract combinations of features and to
        return a flattened array. we assume we will use 0..1 pixel range
        Args:
            imgs: the images to iterate through
            cspace: the color space to use
            spatial_size: the size of spatial binning
            hist_bins: number of bins for color histogram
            pix_per_cell, cell_per_block, hog_channel: parameters for hog features
            spatial_feat, hist_feat, hog_feat: which features to extract
        Returns
            a features array with features for each image
    """
    # initalise a color conversion dictionary
    color_list = {
        'RGB': cv2.COLOR_BGR2RGB,
        'HSV': cv2.COLOR_BGR2HSV,
        'LUV': cv2.COLOR_BGR2LUV,
        'HLS': cv2.COLOR_BGR2HLS,
        'YUV': cv2.COLOR_BGR2YUV,
        'YCrCb': cv2.COLOR_BGR2YCrCb
    }
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # initalise a feature array for the image
        file_features = []
        # Read in each one by one and rescale to 0..1
        image = cv2.imread(file)
        image = image.astype(np.float32)/255
        # apply color conversion
        if cspace in color_list:
            feature_image = cv2.cvtColor(image, color_list[cspace])
        else:
            # bad color space passed, use RGB
            feature_image = cv2.cvtColor(image, color_list['RGB'])

        # extract features if flags are true
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features