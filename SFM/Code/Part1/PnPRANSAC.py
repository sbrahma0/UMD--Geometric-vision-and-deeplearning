#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import numpy as np
import random
from LinearPnP import *

def checkPmatrix(x1, x2, P): 

	reprojection = (x1[0] - (np.matmul(P[0, :], x2))/(np.matmul(P[2, :], x2)))**2 + (x1[1] - (np.matmul(P[1, :], x2))/(np.matmul(P[2, :], x2)))**2
	#print(abs(reprojection))
	return abs(reprojection)

def PnPRANSAC(inliers, traingulatedPoints, K):
	
	noOfInliers = 0
	inlier1 = [] # Variable for storing all the inliers features from the image frame
	inlier2 = [] # Variable for storing all the inliers features from the world frame
	# RANSAC Algorithm
	for i in range(0, 1000): # 1000 iterations for RANSAC 
		count = 0
		sixpoint = []
		goodFeatures1 = [] # Variable for storing six random points from the image frame
		goodFeatures2 = [] # Variable for storing corresponding six random points from the world frame
		tempfeature1 = [] 
		tempfeature2 = []

		while(True): # Loop runs while we do not get eight distinct random points
			num = random.randint(0, len(inliers)-1)
			if num not in sixpoint:
				sixpoint.append(num)
			if len(sixpoint) == 6:
				break

		for point in sixpoint: # Looping over six random points
			goodFeatures1.append(inliers[point]) 
			goodFeatures2.append(traingulatedPoints[point])
		# Computing Projection Matrix from world frame to the image frame using PnP
		newRmat, newCmat = LinearPnP(goodFeatures1, goodFeatures2, K)
		projectionMatrix = np.matmul(K, newRmat)
		temp =  np.hstack((np.eye(3), -newCmat))
		projectionMatrix = np.matmul(projectionMatrix, temp)

		for number in range(0, len(inliers)):
			# If Reprojection Error is less than threshold (0.01) then it is considered as Inlier
			if checkPmatrix(inliers[number], traingulatedPoints[number].reshape(4, 1), projectionMatrix) < 100:
				count = count + 1 
				tempfeature1.append(inliers[number])
				tempfeature2.append(traingulatedPoints[number])

		if count > noOfInliers:
			bestPoints = sixpoint
			noOfInliers = count
			finalR = newRmat
			finalC = newCmat
			inlier1 = tempfeature1
			inlier2 = tempfeature2

	return inlier1, inlier2, finalR, finalC
