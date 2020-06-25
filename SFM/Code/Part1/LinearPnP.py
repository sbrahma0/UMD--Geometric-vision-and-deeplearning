#!/usr/bin/env python

import numpy as np

def LinearPnP(imagePts, worldPts, K):
	
	for i in range(0, len(imagePts)):

		ucoor, vcoor, wcoor = np.matmul(np.linalg.inv(K), np.array([imagePts[i][0], imagePts[i][1], 1]))
		ucoor = ucoor/wcoor
		vcoor = vcoor/wcoor
		'''
		temp = np.array([worldPts[i][0], worldPts[i][1], worldPts[i][2], 1, 0, 0, 0, 0, -worldPts[i][0]*ucoor, -worldPts[i][1]*ucoor, -worldPts[i][2]*ucoor, -ucoor])
		temp = np.vstack((temp, np.array([0, 0, 0, 0, worldPts[i][0], worldPts[i][1], worldPts[i][2], 1, -worldPts[i][0]*vcoor, -worldPts[i][1]*vcoor, -worldPts[i][2]*vcoor, -vcoor])))

		if i == 0:
			A = temp
		else:
			A = np.vstack((A, temp))
	
		'''
		temp = np.array([0, 0, 0, 0, -worldPts[i][0], -worldPts[i][1], -worldPts[i][2], -1, vcoor*worldPts[i][0], vcoor*worldPts[i][1], vcoor*worldPts[i][2], vcoor])
		
		temp = np.vstack((temp, np.array([worldPts[i][0], worldPts[i][1], worldPts[i][2], 1, 0, 0, 0, 0, -ucoor*worldPts[i][0], -ucoor*worldPts[i][1], -ucoor*worldPts[i][2], -ucoor])))
			
		temp = np.vstack((temp, np.array([-vcoor*worldPts[i][0], -vcoor*worldPts[i][1], -vcoor*worldPts[i][2], -vcoor, ucoor*worldPts[i][0], ucoor*worldPts[i][1], ucoor*worldPts[i][2], ucoor, 0, 0, 0, 0])))

		if i == 0:
			A = temp
		else:
			A = np.vstack((A, temp))
		
		
	# For Rotation Matrix
	u, s, v = np.linalg.svd(A) 
	res = v[-1]
	res = res.reshape(3, 4)
	
	R = res[:, 0:3]
	u, s, v = np.linalg.svd(R)
	R = np.matmul(u, v)
	T = res[:, 3]

	if np.linalg.det(R) < 0:
		R = -R
		T = -T
		
	C = -np.matmul(R.T, T)
	C = C.reshape(3, 1)
	# For translation Matrix
	#A = -(R.T)
	#C = np.linalg.inv(np.matmul(A.T, A))
	#C = np.matmul(-R, res[:, 3])
	#C = np.matmul(C, res[:, 3])
	#C = C.reshape(3, 1)

	return R, C	
