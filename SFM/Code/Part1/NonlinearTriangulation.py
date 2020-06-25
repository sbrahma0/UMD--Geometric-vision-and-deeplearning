#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import numpy as np
import math 
from scipy.optimize import least_squares
from LinearTriangulation import *

def NonLinearOptimization(param, P1, P2, pt1, pt2):
	
	data = param.reshape(3,1)
	data = np.vstack((data, [1]))
	result = (pt1[0] - (np.matmul(P1[0, :], data))/(np.matmul(P1[2, :], data)))**2 + (pt1[1] - (np.matmul(P1[1, :], data))/(np.matmul(P1[2, :], data)))**2 + (pt2[0] - (np.matmul(P1[0, :], data))/(np.matmul(P1[2, :], data)))**2 + (pt2[1] - (np.matmul(P1[1, :], data))/(np.matmul(P1[2, :], data)))**2
	return result

def NonlinearTriangulation(P1, P2, inliers1, inliers2):
	
	points = []

	for i in range(0, len(inliers1)):

		temp = list(LinearTriangulation(P1, P2, inliers1[i], inliers2[i]))
		initial_guess = [temp[0][0], temp[1][0], temp[2][0]]
		res = least_squares(NonLinearOptimization, x0=np.array(initial_guess), args=(P1, P2, inliers1[i], inliers2[i]))
		optimize_pt = res.x
		points.append(np.array([optimize_pt[0], optimize_pt[1], optimize_pt[2], 1]))
	
	return points
