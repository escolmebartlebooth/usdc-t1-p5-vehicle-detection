# Vehicle Detection

Author: david escolme
Date: April 2018

## Preamble

The goal was to write a software pipeline to detect vehicles in a video stream. This detection pipeline to implement various feature engineering techniques, a machine learning supervised learning algorithm and ideally to augment the video stream output with the previous lane detection software and to make the pipeline real-time (the latter being stretch targets)

---

The specific goals / steps of this project were the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, to also apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Remembering for the first two steps to normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Pre-requisites

The python environment used for this project can be found at: https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md

Links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples used to train the classifier.

The example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

An optional data source is available with the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment the training data.

[//]: # (Image References)

[image1a]: ./output_images/spatial_bin.png "Spatial Binning"
[image1b]: ./output_images/color_hist.png "Color Histogram"
[image1c]: ./output_images/hog_features.png "hog features"

The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

## Implementation

### Code

+ feature_extraction.py: Contains functions to extract features from images for hog, spatial binning and color histograms
+ feature_extraction.ipynb: notebook visualisation the facets of the feature extraction functions across different parameters and color spaces
+ model_building.ipynb: notebook that fits different models to the training data and tries to tune each to establish a best fit model which can then be saved and reused for the prediction parameters

### Data Preparation and Feature Extraction

The training data was 2 sets of png 64x64 images. Separated into vehicles and non-vehicles.

The images were always rescaled to 0..1 when opened and so any test data would need to also be rescaled to these pixel values when using the trained model for prediction.

As noted in the course notes, matplotlib and opencv have different methods for reading images depending on their type, so it was important to understand when implicit and explicit conversion to 0..1 was needed. For this project, opencv was the sole library used for reading images, which necessitated explicit scaling

Finally, opencv opens images as BGR not RGB by default, this required color space conversion to be BGR2XXX not RGB2XXX

3 main functions were explored for feature extraction:

+ Spatial Binning
+ Color Histogram
+ Histogram of Oriented Gradients (HOG)

The functions for extraction are found in feature_extraction.py and they are visualised and explored in feature_extraction.ipynb

#### Spatial Binning

The idea behind spatial binning is to take an input image and downsize it and then for each color channel to create a concatenated flattened array of pixel values.

This reduces the number of pixels in the image but retains - in theory - enough information to enable the features that are generated from the downsized and flattened image to be used to differentiate objects in a machine learning algorithm.

In the image below, run across a 64x64 RGB png image using a size of 32x32, the car and non-car images can still be deciphered by eye and the flattened array for each, in this case, shows a distinction between the 2 images:

![alt text][image1a]

It's possible to use different color spaces (will generate different patterns of pixel intensity) and different image sizes (will generate less output features as the size is reduced).

By eye, it looked like using the YCrCb color space with a 32x32 size might be interesting but tuning would probably be best done by feeding various feature combinations into the machine learning model to trade off model accuracy and training time.

#### Color Histogram

The color histogram also generates a flattened array across all image channels fed into the function however in this function, the output is a histogram of pixel intensities in each channel.

The tuning parameters for this function are the color space of the image and the number of bins to generate the histogram across. The bin range should relfect the extent of the image pixels.

The idea behind using features from this function is that the color intensities of a car are likely to be different to the intensities of a non-car image. This can be seen in the image below, where 32 bins were applied across a YUV color space image:

![alt text][image1b]

Different color spaces have different histogram signatures and changing the bin size affects the number of discrete features extracted.

#### HOG Features

references:
+ http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
+ https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
+ https://www.learnopencv.com/histogram-of-oriented-gradients/

From the wiki page: _The essential thought behind the histogram of oriented gradients descriptor is that local object appearance and shape within an image can be described by the distribution of intensity gradients or edge directions._

The algorithm has 3 prime tuning parameters:
+ orientations: the number of histograms to generate over the range of gradient angles
+ pixels per cell: the number of pixels which make up a cell.
+ cells per block: the number of cells which make up a block.

The visualisation below shows that for a YCrCb color space image using 8 pixels per cell and 2 cells per block and 9 orientations, the hog features extracted show a different pattern for the car versus the noncar image.

![alt text][image1c]

#### feature extraction summary

Tuning across different parameter combinations and color spaces allows a subjective opinion to be made as to which combinations work best. As a starting point for model building, it looked like using all 3 techniques with the following parameters would be a good bet:

color_space: YCrCb
image size: 64x64 (as training data is already this size)
pixel range: (0..1)

+ spatial binning size: 32x32
+ color histogram bins: 16
+ hog features parameters: all color channels, 9 orientations, 8 pixels per cell, 2 cells per block

A function to iterate over the training data is included which creates a single array with each image's feature set for the extraction parameters above. For the training data we end up with 17760 training images, 8792 of which are cars and a feature vector per image of 8412 features.

The extracted features were saved to data file so they could be loaded into model building without having to regenerate them.

### Model Building, Tuning, Selection and Saving

#### Model setup

The code for the model building can be found in the model_building.ipynb notebook.

#### Initial modelling

The saved training data from the feature_extraction notebook was loaded into memory and then 3 functions were created:
+ split and scale: to create a random shuffled train and test data set and to scale the training data to have zero mean and unit variance
+ build model: simply calling the model's fit method
+ test model: calling the fit model's score method

The 3 classifiers chosen for analysis were:
+ Support Vector Machine (SVM): default settings showed an accuracy of 97.86%. Training time was quite long (using the pre-canned linearSVC was much quicker)
+ Gaussian Bayes: default setting showed an accuracy of 81.64%.
+ Decision Tree: the slowest to train but very quick to predict. Accuracy for the default settings was 94.76%

On that basis, i chose the decision tree and svm to be the models to take forward into hyper-parameter tuning.

#### hyper-parameter tuning

The basis of hyper-parameter tuning is to cycle through combinations of parameters for each model. This is achieved by using a library function called GridSearchCV. This uses 5-fold cross-validation on each model parameter set and then calculates the best estimator from the score function used for each model.

For SVM: We can tune the kernel, c and gamma parameters giving us a total of 16 combinations (more could be tried)...

For Decision Tree: We can tune the criterion, max depth and min samples split parameters giving us 18 combinations....

#### best model

The best model after tuning was found to be ... with the following parameters:
+
+

Using pickle, the best model was saved along with the feature extraction parameters and the standard scalar so that the prediction pipeline could recreate the image pre-processing and feature extraction approach, scale the pipeline images to the same scheme as the trained model and finally re-use the classifier's prediction method to predict which class the pipeline image belongs to

### Classifying Test Images

A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen.

Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)

### Video Implementation

The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video.

A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur.

## Discussion

Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.

## Optional Targets

A stand out submission for this project will be a pipeline that runs in near real time (at least several frames per second on a good laptop) and does a great job of identifying and tracking vehicles in the frame with a minimum of false positives. As an optional challenge, combine this vehicle detection pipeline with the lane finding implementation from the last project! As an additional optional challenge, record your own video and run your pipeline on it to detect vehicles under different conditions.