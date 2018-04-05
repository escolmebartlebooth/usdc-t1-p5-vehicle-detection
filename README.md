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

[image1]: ./output_images/undistort_output.png "Calibration"

The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

## Implementation

### Code

+ feature_extraction.py: Contains functions to extract features from images for hog, spatial binning and color histograms
+ feature_extraction.ipynb: notebook visualisation the facets of the feature extraction functions across different parameters and color spaces

### Data Preparation and Feature Extraction

The training data was 2 sets of png 64x64 images. Separated into vehicles and non-vehicles.

The images were always rescaled to 0..1 when opened and so any test data would need to also be rescaled to these pixel values when using the trained model for prediction.

As noted in the course notes, matplotlib and opencv have different methods for reading images depending on their type, so it was important to understand when implicit and explicit conversion to 0..1 was needed. For this project, opencv was the sole library used for reading images, which necessitated explicit scaling

Finally, opencv opens images as BGR not RGB by default, this required color space conversion to be BGR2XXX not RGB2XXX

3 main functions were explored for feature extraction:

+ Spatial Binning
+ Color Histogram
+ Histogram of Oriented Gradients (HOG)

#### Spatial Binning

#### Color Histogram

#### HOG Features

https://www.learnopencv.com/histogram-of-oriented-gradients/


Each of the extraction techniques, using the tuned parameters, were taken forward into the model building and tuning phases.

### Model Building, Tuning, Selection and Saving

The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier.

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