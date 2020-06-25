#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy
from GetInlierRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *
from LinearTriangulation import *
from NonlinearTriangulation import *
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *

path = '../../Data/'
calibrationMatrix = np.array([[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]])
plottingPoints = []
mainRlist = []
mainClist = []


for i in range(1, 6):

	matchFilePath = path + 'matching' + str(i) + '.txt'
	print(matchFilePath)
	matchFile = open(matchFilePath, 'r')
	data = matchFile.readlines()
		
	features1 = []
	features2 = []
	for line in data:
		line = line.strip()
		line = line.split(' ')
		if line[0] != 'nFeatures:':
			line = map(float, line)
			if i+1 in line[1:]:
				index = line[1:].index(i+1) + 1					
				features1.append([line[4], line[5]])
				features2.append([line[index+1], line[index+2]])

	if i == 1:

		F, inliers1, inliers2 = GetInlierRANSAC(features1, features2)
		print(len(inliers1))
		E = EssentialMatrixFromFundamentalMatrix(calibrationMatrix, F)
		rotationMats, transVecs = ExtractCameraPose(E)
		mainR, mainC, linearTraingPts = DisambiguateCameraPose(calibrationMatrix, rotationMats, transVecs, inliers1, inliers2)
		H1 = np.identity(4)		
		mainRlist.append(np.eye(3))
		mainRlist.append(mainR)
		mainClist.append(H1[0:3, 3])
		mainClist.append(mainC)
		print(mainR, mainC)
		H1 = np.identity(4)
		P1 = np.matmul(np.matmul(calibrationMatrix, np.identity(3)), H1[0:3, :])
		P2 = np.matmul(calibrationMatrix, mainR)
		temp =  np.hstack((np.eye(3), -mainC))
		P2 = np.matmul(P2,temp)
		nonLinearTraingPts = NonlinearTriangulation(P1, P2, inliers1, inliers2)

		for m in range(0, len(nonLinearTraingPts)):
			plt.scatter(float(linearTraingPts[m][0][0]), float(linearTraingPts[m][2][0]), color='r')
			plt.scatter(float(nonLinearTraingPts[m][0]), float(nonLinearTraingPts[m][2]), color='b')
			plottingPoints.append(nonLinearTraingPts[m])
		plt.show()
		P1 = copy.deepcopy(P2)

	else:
		linearTraingPts = []
		P2 = np.matmul(calibrationMatrix, correctedR)
		temp =  np.hstack((np.eye(3), -correctedC))
		P2 = np.matmul(P2, temp)

		#_, inliers1, inliers2 = GetInlierRANSAC(features1, features2)
		nonLinearTraingPts = NonlinearTriangulation(P1, P2, features1, features2)
		for m in range(0, len(plottingPoints)):
			plt.scatter(float(plottingPoints[m][0]), float(plottingPoints[m][2]), color='r')
		for m in range(0, len(nonLinearTraingPts)):
			plt.scatter(float(nonLinearTraingPts[m][0]), float(nonLinearTraingPts[m][2]), color='b')
			plottingPoints.append(nonLinearTraingPts[m])
		plt.show()
		inliers1 = copy.deepcopy(features1)
		P1 = copy.deepcopy(P2)

	if i == 5:
		break

	step = 0
	features4 = []
	threedFeatures = []
	for line in data:
		line = line.strip()
		line = line.split(' ')
		if line[0] != 'nFeatures:':
			line = map(float, line)
			if [line[4], line[5]] == inliers1[step]:
				if i+2 in line[1:]:
					index = line[1:].index(i+2) + 1
					features4.append([line[index+1], line[index+2]])
					threedFeatures.append(nonLinearTraingPts[step])
				step += 1
		if step == len(inliers1) - 1:
			break
		
	print(len(features4))
	pnpinliers3, pnpinliers4, pnpR, pnpC = PnPRANSAC(features4, threedFeatures, calibrationMatrix)
	print(pnpR, pnpC, len(pnpinliers3))
	correctedR, correctedC = NonlinearPnP(pnpinliers3, pnpinliers4, calibrationMatrix, pnpR, pnpC)
	print(correctedR, correctedC)
	mainRlist.append(correctedR)
	mainClist.append(correctedC)

for i in range(0, len(plottingPoints)):
	plt.scatter(float(plottingPoints[i][0]), float(plottingPoints[i][2]), color='r')
plt.show()

Vmatrix, vpoints = BuildVisibilityMatrix(mainRlist, mainClist, calibrationMatrix)
print Vmatrix, Vmatrix.shape
