#!/usr/bin/env python

import numpy as np

def LinearTriangulation(m1, m2, point1, point2):
	# Skew Symmetric Matrix of point1
	oldx = np.array([[0, -1, point1[1]], [1, 0, -point1[0]], [-point1[1], point1[0], 0]]) 
	# Skew Symmetric Matrix of point2
	oldxdash = np.array([[0, -1, point2[1]], [1, 0, -point2[0]], [-point2[1], point2[0], 0]])

	A1 = np.matmul(oldx, m1)
	A2 = np.matmul(oldxdash, m2)
	A = np.vstack((A1, A2)) # Ax = 0

	u, s, v = np.linalg.svd(A)
	new1X = v[-1]
	new1X = new1X/new1X[3]
	new1X = new1X.reshape((4,1))

	return new1X[0:3].reshape((3,1))

