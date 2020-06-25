#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import numpy as np
import math 
import matplotlib.pyplot as plt
from LinearTriangulation import *

def rotationMatrixToEulerAngles(R) :
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

def DisambiguateCameraPose(K, Rlist, Clist, inliers1, inliers2):

	check = 0
	P1 = np.matmul(K, np.identity(4)[0:3, :]) # current camera pose is always considered as a identity matrix
	colour = ['r', 'b', 'g', 'y']
	for index in range(0, len(Rlist)): # Looping over all the rotation matrices
		angles = rotationMatrixToEulerAngles(Rlist[index]) # Determining the angles of the rotation matrix
		pts = []
		# If the rotation of x and z axis are within the -50 to 50 degrees then it is considered down in the pipeline 
		if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50: 
			count = 0
			P2 = np.matmul(K,Rlist[index])
			temp =  np.hstack((np.eye(3),-Clist[index]))
			P2 = np.matmul(P2,temp)  # New camera Pose
 		
			for i in range(0, len(inliers1)): # Looping over all the inliers
				temp1x = LinearTriangulation(P1, P2, inliers1[i], inliers2[i]) # Triangulating all the inliers
				pts.append(temp1x)
				plt.scatter(temp1x[0][0], temp1x[2][0], color=colour[index])
				thirdrow = Rlist[index][2,:].reshape((1,3)) 
				if np.squeeze(np.matmul(thirdrow, (temp1x - Clist[index]))) > 0: # If the depth of the triangulated point is positive
					count = count + 1 
				#print(count)
			if count > check: 
				maincolor = colour[index]
				check = count
				mainc = Clist[index]
				mainr = Rlist[index]
				triangulatedPts = pts

	plt.show()

	for pts in triangulatedPts:
		plt.scatter(pts[0][0], pts[2][0], color=maincolor)
	plt.show()
	#if mainc[2] > 0:
	#	mainc = -mainc

	#print('mainangle', rotationMatrixToEulerAngles(mainr))
	return mainr, mainc, triangulatedPts
