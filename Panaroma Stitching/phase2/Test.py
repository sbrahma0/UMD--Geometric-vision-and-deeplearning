#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import SupervisedModel, UnsupervisedModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(PatchPath, GtPath):
	"""
	Inputs: 
	BasePath - Path to images
	Outputs:
	ImageSize - Size of the Image
	DataPath - Paths of all images where testing will be run on
	"""   
	# Image Input Shape
	ImageSize = [128, 128, 2]
	DataPatchPath = []
	DataGtPath = []
	NumImages = len(glob.glob(PatchPath+'*.png'))
	SkipFactor = 1
	for count in range(1, NumImages+1, SkipFactor):
		DataPatchPath.append(PatchPath + str(count) + '.png')
		DataGtPath.append(GtPath + str(count) + '.png')

	return ImageSize, DataPatchPath, DataGtPath
    
def ReadImages(ImageSize, DataPatchPath, DataGtPath):
	"""
	Inputs: 
	ImageSize - Size of the Image
	DataPath - Paths of all images where testing will be run on
	Outputs:
	I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
	I1 - Original I1 image for visualization purposes only
	"""
	ImageName1 = DataPatchPath
	ImageName2 = DataGtPath

	I1 = cv2.imread(ImageName1)
	I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2 = cv2.imread(ImageName2)
	I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

	if(I1 is None):
		# OpenCV returns empty list if image is not read! 
		print('ERROR: Image I1 cannot be read')
		sys.exit()

	##########################################################################
	# Add any standardization or cropping/resizing if used in Training here!
	##########################################################################

	I1S = iu.StandardizeInputs(np.float32(I1))
	I2S = iu.StandardizeInputs(np.float32(I2))
	imgData = np.dstack((I2S, I1S))
	imgData = np.reshape(imgData, (1, 128, 128, 2))

	return imgData
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPatchPath, DataGtPath, LabelsPathPred):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	ImageSize is the size of the image
	ModelPath - Path to load trained model from
	DataPath - Paths of all images where testing will be run on
	LabelsPathPred - Path to save predictions
	Outputs:
	Predictions written to ./TxtFiles/PredOut.txt
	"""
	Length = ImageSize[0]
	# Predict output with forward pass, MiniBatchSize for Test is 1
	prSoftMaxS = SupervisedModel(ImgPH, ImageSize, 1)
	#prSoftMaxS = UnsupervisedModel(ImgPH, ImageSize, 1)

	# Setup Saver
	Saver = tf.compat.v1.train.Saver()
	with tf.compat.v1.Session() as sess:
		Saver.restore(sess, ModelPath)
		print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
		OutSaveT = open(LabelsPathPred, 'w')

		for count in tqdm(range(np.size(DataPatchPath))):            
			DataPatchPathNow = DataPatchPath[count]
			DataGtPathNow = DataGtPath[count]
			Img = ReadImages(ImageSize, DataPatchPathNow, DataGtPathNow)
			FeedDict = {ImgPH: Img}
			PredT = sess.run(prSoftMaxS, FeedDict)
			print PredT
			PredT = (np.squeeze(PredT)*20).astype(int) 
			print PredT
			OutSaveT.write(str(PredT)+'\n')
		
		OutSaveT.close()


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
		l1 = []
		LabelTest = open(LabelsPathTest, 'r')
		for line in LabelTest.readlines():
			line = line.split(' ')
			l1.append(map(int, line[1:8]))

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
		l2 = []
		LabelPred = open(LabelsPathPred, 'r')
		for line in LabelPred.readlines():
			line = line.split(' ')
			l2.append(map(int, line[1:8]))
        
    return l1, l2

        
def main():
	"""
	Inputs: 
	None
	Outputs:
	Prints out the confusion matrix with accuracy
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ModelPath', dest='ModelPath', default='Checkpoints/Checkpoints14model.ckpt', help='Path to load latest model from, Default:ModelPath')
	Parser.add_argument('--PatchPath', dest='PatchPath', default='../Data/TrainPatch/', help='Path to load images from, Default:BasePath')
	Parser.add_argument('--GtPath', dest='GtPath', default='../Data/TrainGroundTruth/', help='Path to load images from, Default:BasePath')
	Parser.add_argument('--LabelsPath', dest='LabelsPath', default='TxtFiles/TrainLabels.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
	Args = Parser.parse_args()
	ModelPath = Args.ModelPath
	PatchPath = Args.PatchPath
	GtPath = Args.GtPath
	LabelsPath = Args.LabelsPath

	# Setup all needed parameters including file reading
	ImageSize, DataPatchPath, DataGtPath = SetupAll(PatchPath, GtPath)

	# Define PlaceHolder variables for Input and Predicted output
	ImgPH = tf.compat.v1.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], ImageSize[2]))
	LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

	TestOperation(ImgPH, ImageSize, ModelPath, DataPatchPath, DataGtPath, LabelsPathPred)
	# Plot Confusion Matrix
	#LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
	#ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
