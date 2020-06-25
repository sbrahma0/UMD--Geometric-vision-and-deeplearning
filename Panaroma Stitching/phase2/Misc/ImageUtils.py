#!/usr/bin/env python

import cv2
import numpy as np
import sys

sys.dont_write_bytecode = True

def findMax(val):
	return np.max(val)

def findMin(val):
	return np.min(val)

def findMean(val):
	return np.mean(val)

def findStd(val):
	return np.sqrt(np.sum((val - findMean(val))**2)/len(val))

def StandardizeInputs(img):
		
	channel1 = img[:, :]
	#channel2 = img[:, :, 1]
	#channel3 = img[:, :, 2]	

	#Normalization
	if(findMax(channel1) - findMin(channel1)) != 0:
		channel1 = (channel1 - findMin(channel1))/(findMax(channel1) - findMin(channel1))
	#channel2 = (channel2 - findMin(channel2))/(findMax(channel2) - findMin(channel2))
	#channel3 = (channel3 - findMin(channel3))/(findMax(channel3) - findMin(channel3))

	#Standardization
	if findStd(channel1) != 0:
		channel1 = (channel1 - findMean(channel1))/findStd(channel1)
	#channel2 = (channel2 - findMean(channel2))/findStd(channel2)
	#channel3 = (channel3 - findMean(channel3))/findStd(channel3)

	#New Standardize Image
	newImage = np.zeros((channel1.shape[0], channel1.shape[1]))
	newImage = channel1
	#newImage[:, :, 1] = channel2
	#newImage[:, :, 2] = channel3

	return newImage

def main():
	image = cv2.imread('../../CIFAR10/Test/1.png')
	StandardizeInputs(np.float32(image))

if __name__ == '__main__':
	main()
