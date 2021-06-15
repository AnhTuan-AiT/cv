import argparse as ap
import os

import cv2
import joblib
import numpy as np
from scipy.cluster.vq import *

import imutils

# Load the classifier, class names, scaler, number of clusters and vocabulary
classifier, classes_name, stdSlr, num_clusters, vocabulary = joblib.load("bof.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v', "--visualize", action='store_true')
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []

if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print("No such directory {}\nCheck if the file exists".format(test_path))
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths += class_path
else:
    image_paths = [args["image"]]

# print(image_paths)


# Create feature extraction and keypoint detector objects
sift = cv2.SIFT_create()
# des_ext = cv2.SIFT_create()

# List where all the descriptors are stored
img_descriptor_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)

    if im is None:
        print("No such file {}\nCheck if the file exists".format(image_path))
        exit()

    # Visualize keypoints of image.
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # kp = sift.detect(gray, None)
    # img=cv2.drawKeypoints(gray,kp,im,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints.jpg', img)

    # Combine in one method detectAndCompute().
    # keypoints = sift.detect(im)
    # keypoints, des = des_ext.compute(im, keypoints)

    # Here keypoints will be a list of keypoints and des is a numpy array of shape(Number of keypoints)×128
    keypoints, des = sift.detectAndCompute(im, None)
    img_descriptor_list.append((image_path, des))

# print(img_descriptor_list)


# Stack all the descriptors vertically in a numpy array
# descriptors = img_descriptor_list[0][1]
# print(descriptors)
# for image_path, img_des in img_descriptor_list[1:]:
#     descriptors = np.vstack((descriptors, img_des))
# print(descriptors)
#

# test_features is classified feature result of images
test_features = np.zeros((len(image_paths), num_clusters), "float32")

for i in range(len(image_paths)):
    words, distance = vq(img_descriptor_list[i][1], vocabulary)

    for w in words:
        test_features[i][w] += 1

# print(test_features)


# Perform Tf-Idf vectorization
# num_occurences = np.sum((test_features > 0) * 1, axis=0)
# idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * num_occurences + 1)), 'float32')


# Scale the features, std_img_test_features
test_features = stdSlr.transform(test_features)

# Perform the predictions
predictions = [classes_name[i] for i in classifier.predict(test_features)]

# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        cv2.imshow("Image", image)
        cv2.waitKey(3000)
