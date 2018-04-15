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
[image2a]: ./output_images/test1.png "pipeline image 1"
[image2b]: ./output_images/test2.png "pipeline image 2"
[image2c]: ./output_images/test3.png "pipeline image 3"
[image2d]: ./output_images/test4.png "pipeline image 4"
[image2e]: ./output_images/test5.png "pipeline image 5"
[image2f]: ./output_images/test6.png "pipeline image 6"
[video1]: ./project_video_output.mp4 "Video"


## Implementation

### Code

+ feature_extraction.py: Contains functions to extract features from images for hog, spatial binning and color histograms
+ feature_extraction.ipynb: notebook visualisation the facets of the feature extraction functions across different parameters and color spaces
+ model_building.ipynb: notebook that fits different models to the training data and tries to tune each to establish a best fit model which can then be saved and reused for the prediction parameters
+ vehicle_detection.ipynb: notebook where the pipeline was tested on static and video images and where the pipeline class was defined
+ lane_detector.py: re-implementation of the lane detection software for adding lane lines to the video image

### Data Preparation and Feature Extraction

The training data was 2 sets of png 64x64 images. Separated into vehicles and non-vehicles.

The images were always rescaled to 0..1 when opened and so any test data would need to also be rescaled to these pixel values when using the trained model for prediction.

As noted in the course notes, matplotlib and opencv have different methods for reading images depending on their type, so it was important to understand when implicit and explicit conversion to 0..1 was needed. For this project, opencv was the sole library used for reading images during static image processing, which necessitated explicit scaling.

Finally, opencv opens images as BGR not RGB by default, this required color space conversion to be BGR2XXX not RGB2XXX for static images. For the video pipeline, the images are opened to RGB by default.

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
+ hog features parameters: All Hog channels, 9 orientations, 8 pixels per cell, 2 cells per block

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
+ Support Vector Machine (SVM):
    + Linear SVC: default settings showed an accuracy of 98.59%. Quick training (23s) and testing time (0.17s)
    + Non-Linear (rbf): Accuracy: 98.59% Slow training (184s) and testing time (44s)
+ Gaussian Bayes: default setting showed an accuracy of 89.64%.
+ Decision Tree: the slowest to train but very quick to predict. Accuracy for the default settings was 96.14%

On that basis, i chose the decision tree and linear svm to be the models to take forward into hyper-parameter tuning as each had good initial accuracy and both were fast predictors.

#### hyper-parameter tuning

The basis of hyper-parameter tuning is to cycle through combinations of parameters for each model. This is achieved by using a library function called GridSearchCV. This uses 5-fold cross-validation on each model parameter set and then calculates the best estimator from the score function used for each model.

For Linear SVM: We can tune the c parameter. For various values of 'c' the training time is roughly 22s x 5 x n (values of 'c') as GridSearchCV uses 5 fold cross-validation. This gives a total training time of ~400s.

For Decision Tree: We can tune the criterion, max depth and min samples split parameters giving us 18 combinations and an approximate search time of 7 to 8 hours.

#### best model

The best model after tuning was found to be LinearSVC with c = 0.001 and a classification accuracy of 98.68% however I chose to stick with C=1 as i felt that under unseen video images, the classifier might generalise better and detect fewer false positives.

Using pickle, the best model was saved along with the feature extraction parameters and the standard scalar so that the prediction pipeline could recreate the image pre-processing and feature extraction approach, scale the pipeline images to the same scheme as the trained model and finally re-use the classifier's prediction method to predict which class the pipeline image belongs to

### Classifying Test Images

The pipeline used to detect cars on test images (and then video frames) was implemented into a vehicle_detector class which also incorporated the code which allows lane lines to be detected. The code can be found in vehicle_detection.ipynb.

The pipeline for vehicle detection was implemented in the ```process_image``` method, the pipeline being:
+ load the feature extraction parameters, model scalar and model for use in image processing
+ create an instance of the detector class and initialise its parameters from the feature extraction parameters
+ for each image passed to process_image():
+ initialise the image:
    + crop the image to a region of interest - there's little point trying to detect images in the sky so lose the top 40% of the image and bottom 10%
    + performs a color conversion on the image to match the color space used in feature extraction
    + Resize the image based on scale parameter (see later for explanation)
    + Scale the image pixel intensities to 0..1 (this assumes that the input image is scaled 0..255)
+ hog features: hog features are extracted for the entire cropped image created when initialising the image
+ vehicle detection: a sliding window technique is then performed on the initialised image for each scale passed into the detector:
    + the sliding window operates over the equivalent of multiple overlapping 64x64 sub images of the initialised image
    + hog features, spatial binning and color histograms are extracted for each 64x64 window to create a flattened feature vector
    + each window is then processed via the model to scale and then predict whether that window contains a vehicle or not
    + if the window is deemed to contain a vehicle that window is added to a list of 'vehicle windows' for the entire image
    + this process continues for each 64x64 window extracted from the image
    + if more than one scale is passed into the detector, the process is repeated for each scale, where the initial image is scaled by the scale parameter. Scaling allows for the idea that cars vary in size depending on their place in the image and so it makes sense to operate over multiple scales to improve the chances of detecting vehicles of different sizes.
+ heatmap, threshold and draw: once the pipeline is completed a set of boxes for detected images are available for the image. These are processed using a heatmap approach where each image pixel is incremented by 1 for every detected box it falls within. Using a threshold and labels function from sklearn, individual cars can be detected and boxes drawn onto the final output image

For the test images, the various pipeline stages can be seen:

![alt text][image2a]
![alt text][image2b]
![alt text][image2c]
![alt text][image2d]
![alt text][image2e]
![alt text][image2f]

The pipeline will detect false positives - where the postive detection is not for a car. Various options exist to reduce the number of false positives on single images:
+ better feature extraction - looking for the combination of features which create a better dataset for training
+ better training data - providing more data for training and/or accounting for the time-series nature of the data used so that the test/train split is done better
+ a more robust model - further model tuning to improve accuracy

### Video Implementation

The video implementation differs from the single image implementation only in that the position of detected boxes is averaged over a configurable number of frames.

This averaging takes the form of a collection with a maximum length of n into which all detected windows are added. The threshold function is then used to eliminate detections that appear <= a threshold parameter passed to the vehicle detector.


## Discussion

The primary problem for this project was reducing the number of false positives detected by the mode and understanding how each factor in that task interplayed:
+ feature extraction - which features to choose to feed into the model
+ model building - which model to train and tune
+ static image pipeline processing - what thresholds to apply to the heatmap to reduce false positives
+ frame by frame processing - over how many frames and how to integrate detections to remove false positives

The failure modes for this pipeline would be the ability to detect other objects on the road (bicycles, motor-cycles, pedestrians, trucks) as no training data for these images are included in the training data and the model is a single classifier of car / not car.

It looks also as if different lighting / weather conditions would cause failures as the model seemed to work best on clear, well lit and 'clean' road conditions.

A number of potential Improvements would be to:
+ parallel processing of the multiple scales and also the lane detection to achieve near realtime frame processing
+ increase the training data set and account better for the time-series images in the GTI data and also widen the classification to include different types of object
+

