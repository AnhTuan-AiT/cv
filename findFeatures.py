#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# Create feature extraction and keypoint detector objects
sift = cv2.SIFT_create()
des_ext = cv2.SIFT_create()

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = sift.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
num_clusters = 100
vocabulary, variance = kmeans(descriptors, num_clusters, 1)

# Calculate the histogram of features
img_features = np.zeros((len(image_paths), num_clusters), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], vocabulary)
    for w in words:
        img_features[i][w] += 1

# Perform Tf-Idf vectorization
num_occurences = np.sum((img_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * num_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(img_features)
img_features = stdSlr.transform(img_features)

# Train the Linear SVM
classifier = LinearSVC()
classifier.fit(img_features, np.array(image_classes))

# Save the SVM
joblib.dump((classifier, training_names, stdSlr, num_clusters, vocabulary), "bof.pkl", compress=3)
