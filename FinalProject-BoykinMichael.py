#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USAGE
# python knn.py --dataset ../datasets/animals

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import numpy as np
import cv2
import os
import random
import pickle

def load(imagePath_list, method, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []
		
		# loop over the input images
		for (i, imagePath) in enumerate(imagePath_list):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
           
			label = imagePath.split(os.path.sep)[-2]

			# check to see if our preprocessors are not None
			image = cv2.resize(image, (32, 32),interpolation=cv2.INTER_CUBIC)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)

			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagePath_list)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))

# grab the list of images that we'll be describing

print("[INFO] loading images...")

imagePath_list = list(paths.list_images("./RockTypes"))
# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
(data, labels) = load(imagePath_list, method=0, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using random of between
# 60%-80% for training, 10%-20% of testing  and the remainder for validation
data_size_number = round((random.randint(60, 80))/ 100, 2)
test_size_number = round((random.randint(10, 15) / 100), 2)
valid_size_number = round(1 - (test_size_number + data_size_number), 2)
#if our data size happens to be randomly 80, then we'll chose 5% and 15%
if ( valid_size_number == 0.00) :
	valid_size_number = 0.05
	test_size_number = 0.15

print ("\nData size: {:.2f} Test size: {:.2f} Validation size: {:.2f}\n"
	.format(data_size_number, test_size_number, valid_size_number))
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=test_size_number, random_state= 42)

(trainX, validateX, trainY, validateY) = train_test_split(data, labels,
	test_size=valid_size_number, random_state = 42)
pickleSave = False

if( pickleSave == True ) :
	# train and evaluate a k-NN classifier on the raw pixel intensities
	k = [3, 5, 7]
	line = [' ', 'Manhattan method', 'Euclidian method']
	for linDist in range(1, 3):
		for kVal in k:
			print("[INFO] evaluating k-NN classifier for k-value ", kVal, " using ", line[linDist])
			model = KNeighborsClassifier(n_neighbors=kVal, p=linDist)
			model.fit(trainX, trainY)
			print(classification_report(testY, model.predict(testX), target_names=le.classes_))
			print("\n")
	#saves the model to disk
	pickle.dump(model, open('kNNModel.sav', 'wb'))
else:
	model = pickle.load( open('kNNModel.sav', 'rb'))

imagePath_list = list(paths.list_images("./tests"))
(dataTest, labels) = load(imagePath_list, method=1, verbose=500)
dataTest = dataTest.reshape((dataTest.shape[0], 3072))

dataTestGuess = model.predict( dataTest )
print ( le.inverse_transform(dataTestGuess))
