#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

import numpy as np
from NonlinearTriangulation import *

def BuildVisibilityMatrix(Rlist, Clist, K):

	path = '../../Data/'
	visibilityMatrix = []
	points = []
	for i in range(1, 6):
		matchFilePath = path + 'matching' + str(i) + '.txt'
		matchFile = open(matchFilePath, 'r')
		data = matchFile.readlines()

		P1 = np.matmul(K, Rlist[i-1])
		temp =  np.hstack((np.eye(3), -Clist[i-1].reshape(3,1)))
		P1 = np.matmul(P1, temp)

		P2 = np.matmul(K, Rlist[i])
		temp =  np.hstack((np.eye(3), -Clist[i].reshape(3,1)))
		P2 = np.matmul(P2, temp)

		for line in data:
			line = line.strip()
			line = line.split(' ')
			if line[0] != 'nFeatures:':
				line = map(float, line)
				points.append(NonlinearTriangulation(P1, P2, [[line[4], line[5]]], [[line[7], line[8]]]))
				temp2 = []
				for j in range(1, 7):
					if j == i:
						temp2.append(1)
					else:
						if j in line[1:]:
							temp2.append(1)
						else:
							temp2.append(0)
				
				visibilityMatrix.append(temp2)
		
	visibilityMatrix = np.array(visibilityMatrix)
	visibilityMatrix = visibilityMatrix.reshape(len(points), 6)

	return visibilityMatrix, points

