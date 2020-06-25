#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import numpy as np
import random
from EstimateFundamentalMatrix import *

def checkFmatrix(x1,x2,F): 

    x11 = np.array([x1[0],x1[1],1]).T
    x22 = np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))

def GetInlierRANSAC(features1, features2):

	noOfInliers = 0
	inlier1 = [] # Variable for storing all the inliers features from the current frame
	inlier2 = [] # Variable for storing all the inliers features from the next frame
	# RANSAC Algorithm
	for i in range(0, 5000): # 5000 iterations for RANSAC 
		count = 0

		eightpoint = []
		goodFeatures1 = [] # Variable for storing eight random points from the current frame
		goodFeatures2 = [] # Variable for storing corresponding eight random points from the next frame
		tempfeature1 = [] 
		tempfeature2 = []
		
		while(True): # Loop runs while we do not get eight distinct random points
			num = random.randint(0, len(features1)-1)
			if num not in eightpoint:
				eightpoint.append(num)
			if len(eightpoint) == 8:
				break
		
		for point in eightpoint: # Looping over eight random points
			goodFeatures1.append([features1[point][0], features1[point][1]]) 
			goodFeatures2.append([features2[point][0], features2[point][1]])

		# Computing Fundamentals Matrix from current frame to next frame
		FundMatrix = EstimateFundamentalMatrix(goodFeatures1, goodFeatures2)

		for number in range(0, len(features1)):
			# If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
			if checkFmatrix(features1[number], features2[number], FundMatrix) < 0.005:
				count = count + 1 
				tempfeature1.append(features1[number])
				tempfeature2.append(features2[number])

		if count > noOfInliers: 
			bestPoints = eightpoint
			noOfInliers = count
			finalFundMatrix = FundMatrix
			inlier1 = tempfeature1
			inlier2 = tempfeature2

	return FundMatrix, inlier1, inlier2
