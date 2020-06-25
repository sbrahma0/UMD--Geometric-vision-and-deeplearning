#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
import glob
import copy
import random

# Add any python libraries here

def AdaptiveNonMaximalSuppression(image, cornersList, number_corners):

	l,_,n = cornersList.shape
	r = np.full((l, 3), np.inf)
	ed = 0
	for i in range(l):
		xi = cornersList[i, :, 0]
		yi = cornersList[i, :, 1]

		for j in range(l):
			xj = cornersList[j, :, 0]
			yj = cornersList[j, :, 1]
	
			if (image[yi, xi] < image[yj, xj]):
				ed = (xj-xi)**2 + (yj-yi)**2
			if ed < r[i,0]:
				r[i,0] = ed
				r[i,1] = xi
				r[i,2] = yi

	r = sorted(r, key=lambda r_entry: r_entry[0])
	r = np.asarray(r)
	return (r[:number_corners,:])

def generateFeatures(corners, image):
		
	features = []

	for i in corners:
		_, x3, y3 = i.ravel()
		
		if (x3 - 20 > 0) and (y3 - 20 > 0) and ((x3 + 20) < image.shape[1]) and ((y3 + 20) < image.shape[0]):
			
			# For every corner generate a 60x60 patch considering corner as the center
			croppedImage = image[(int(y3) - 20):(int(y3) + 20), (int(x3) - 20):(int(x3) + 20)]
			croppedImage = cv.GaussianBlur(croppedImage, (5,5), 0)
			# Resize 60x60 patch to 8x8			
			resizedImage = cv.resize(croppedImage, (8, 8))
			# Convert 8x8 patch to 1x64 feature vector
			resizedImageVector = np.reshape(resizedImage, (1, 64))
			# Standardize the feature vector
			mean = np.mean(resizedImageVector)
			std = np.sqrt(np.sum((resizedImageVector - mean)**2)/len(resizedImageVector))
			resizedImageVector = (resizedImageVector - mean)/(std)		
			features.append([int(x3), int(y3), resizedImageVector])

	'''
	temp2 = copy.deepcopy(image)
	I = np.zeros((100, 1345), dtype=np.uint8)
	temp = cv.resize(croppedImage, (8, 8))
	temp = np.reshape(temp, (1, 64))
	j = 0
	for i in temp[0]:
		I[0:100, j:j+20] = np.ones((100, 20))*i
		j = j + 20 + 1
	
	cv.line(temp2, (int(x3-20), int(y3-20)), (int(x3+20), int(y3-20)), (255, 255, 255), 5)
	cv.line(temp2, (int(x3+20), int(y3-20)), (int(x3+20), int(y3+20)), (255, 255, 255), 5)
	cv.line(temp2, (int(x3+20), int(y3+20)), (int(x3-20), int(y3+20)), (255, 255, 255), 5)
	cv.line(temp2, (int(x3-20), int(y3+20)), (int(x3-20), int(y3-20)), (255, 255, 255), 5)
	fig = plt.figure(1)
	fig.add_subplot(1, 3, 1)
	plt.imshow(temp2, cmap='gray')
	plt.axis('off')
	fig.add_subplot(1, 3, 2)
	plt.imshow(croppedImage, cmap='gray')
	plt.axis('off')
	fig.add_subplot(1, 3, 3)
	plt.imshow(I, cmap='gray')
	plt.axis('off')
	plt.show()
	'''

	return features


def main():

	images_col = []
	#Read a set of images for Panorama stitching
		
	filenames = glob.glob("../Data/Train/Set1/*.jpg")
	filenames.sort()

	for img in filenames:

		imgRGB = cv.imread(img)
		imageResized = cv.resize(imgRGB, (400, 400))
		images_col.append(imageResized)

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	h1 = np.identity(3)	# Initialization of the homography matrix for the newImage
	h2 = np.identity(3) # Initialization of the homography matrix for the lastImage
	final = np.zeros((2000, 2000, 3), dtype = np.uint8) # Initialization of the final canvas

	for index in range(len(images_col) - 1):
	
		# Image upon which newImage will be stitched using homography
		lastImage = images_col[index] 
		# Image to be stitched using homography		
		newImage = images_col[index + 1]
		lastGray = cv.cvtColor(lastImage, cv.COLOR_BGR2GRAY)
		newGray = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)

		# Shi-Thomsi Features for the lastImage
		corners1 = cv.goodFeaturesToTrack(lastGray, 500, 0.01, 10)
		corners1 = np.int0(corners1)

		# Shi-Thomsi Features for the newImage
		corners2 = cv.goodFeaturesToTrack(newGray, 500, 0.01, 10)
		corners2 = np.int0(corners2)

		"""
		plotImage1 = copy.deepcopy(lastImage)	
		plotImage2 = copy.deepcopy(newImage)	
		for i in corners1:
			x1, y1 = i.ravel()
			cv.circle(plotImage1, (x1, y1), 2, (0, 0, 255), -1)
	
		for i in corners2:
			x1, y1 = i.ravel()
			cv.circle(plotImage2, (x1, y1), 2, (0, 0, 255), -1)

		cv.imshow("lastImage", plotImage1)
		cv.imshow("newImage", plotImage2)
		cv.waitKey(0)
		cv.destroyAllWindows()	
		"""
		"""
		Perform ANMS: Adaptive Non-Maximal Suppression
		Save ANMS output as anms.png
		"""
		# Adaptive Non-Maximum Suppression on the lastImage
		uniformCorners1 = AdaptiveNonMaximalSuppression(lastGray, corners1, 300)
		# Adaptive Non-Maximum Suppression on the newImage		
		uniformCorners2 = AdaptiveNonMaximalSuppression(newGray, corners2, 300)
		
		"""
		plotImage3 = copy.deepcopy(lastImage)
		plotImage4 = copy.deepcopy(newImage)
		
		for i in uniformCorners1:
			_, x2, y2 = i.ravel()
			cv.circle(plotImage3, (int(x2), int(y2)), 2, (0, 0, 255), -1)

		for i in uniformCorners2:
			_, x2, y2 = i.ravel()
			cv.circle(plotImage4, (int(x2), int(y2)), 2, (0, 0, 255), -1)
	
		cv.imshow("lastImageUniform", plotImage3)
		cv.imshow("newImageUnifrom", plotImage4)
		cv.waitKey(0)
		cv.destroyAllWindows()	
		"""
		"""
		Feature Descriptors
		Save Feature Descriptor output as FD.png
		"""
		# Feature Generation for the lastImage
		features1 = generateFeatures(uniformCorners1, lastGray)
		# Feature Generation for the newImage
		features2 = generateFeatures(uniformCorners2, newGray)

		"""
		Feature Matching
		Save Feature Matching output as matching.png
		"""
		keypoint1 = [] # List Initialization for storing important feature coordinates for lastImage
		keypoint2 = [] # List Initialization for storing important feature coordinates for newImage

		for i in range(len(features1)): # Looping over all the features form the lastImage
			squaredDist = [] # List Initialization for storing squared distance between the features
			for j in range(len(features2)): # Looping over all the features form the newImage
				temp = np.sum((features1[i][2] - features2[j][2])**2) # Calculating squared distance
				squaredDist.append([i, j, temp])
		
			# Sorting squared distance list in ascending order
			squaredDist.sort(key = lambda squaredDist: squaredDist[2])
			# Calculating ratio of smallest distance to the second smallest distance 
			ratio = squaredDist[0][2]/squaredDist[1][2]
			if ratio < 0.7: # If the above ratio is less than 0.7(hyperparameter)
				# Store the corresponding feature coordinates  
				keypoint1.append([features1[squaredDist[0][0]][0], features1[squaredDist[0][0]][1]])
				keypoint2.append([features2[squaredDist[0][1]][0], features2[squaredDist[0][1]][1]])
	
		"""
		plotImage5 = np.hstack((lastImage, newImage))	
		for i in range(len(keypoint1)):
			cv.circle(plotImage5, (keypoint1[i][0], keypoint1[i][1]), 4, (255, 0, 0), 2) 
			cv.circle(plotImage5, (keypoint2[i][0]+lastGray.shape[1], keypoint2[i][1]), 4, (255, 0, 0), 2) 
			cv.line(plotImage5, (keypoint1[i][0], keypoint1[i][1]), (keypoint2[i][0]+lastGray.shape[1], keypoint2[i][1]), (0, 0, 255), 1)

		cv.imshow('FeatureMatching', plotImage5)
		cv.imshow('temp', temp)
		cv.waitKey(0)
		cv.destroyAllWindows()
		"""
		"""
		Refine: RANSAC, Estimate Homography
		"""
		print "Length of features: " + str(len(keypoint1))
		if len(keypoint1) <= 4:
			print "Stitching Not Possible due to less overlapping"
			break
		inliers = 0 # Initializing variable inliers to zero
		maxInliers = 0 # Initializing variable maxInliers to zero
		maxList = [] # List for storing inliers corresponding to the maxInliers variable
		maxH = np.identity(3) # Matrix for storing homography corresponding to the maxInliers variable
		percent = 0.95 # Initial Inliers percentage requirement
		itr = 1 # Variable for counting the number of iterations

		while(1):

			# If total number of iteration is 1000 then returns the maximum number of inliers found upto 1000 iterations
			if itr == 1000:
				inliers = maxInliers
				inliersList = maxList
				H = maxH
				break

			# Every 200 iterations reduce the inliers percentage requirement by 5%
			if itr%200 == 0:
				percent = percent - 0.05
		
			# If the number of inliers are greater than the requirement
			if(inliers >= int(len(keypoint1)*percent)):
				# Check if the maxinliers found up until now is greater than the current number
				# If yes then returns the maxliers found earlier
				if maxInliers > inliers:
					inliers = maxInliers
					inliersList = maxList
					H = maxH
				break

			inliersList = [] # Temporary list for storing inliers in each iteration
			inliers = 0	# Assigning zero to the number of inliers in each iteration
			randomNum = [] # Temporary list for storing four random numbers in each iteration
			while len(randomNum) != 4:
				temp = random.randrange(0, len(keypoint1))
				if temp not in randomNum:
					randomNum.append(temp)
	
			points1 = [] # Temporary list for storing four random coordinates form the lastImage in each iteration
			points2 = [] # Temporary list for storing four random coordinates form the newImage in each iteration
			for num in randomNum:
				points1.append(keypoint1[num])
				points2.append(keypoint2[num])

			# Calculating homography between the four random points from newImage to lastImage in each iteration
			H, status = cv.findHomography(np.array(points2), np.array(points1))
			if status[0][0] != 0: # If we get a homography matrix 
				# Looping through all the keypoint features from newImage 
				for m in range(len(keypoint2)):
					# Calculating warped point from newImage to lastImage using the above homography matrix in each iteration
					newPoint = np.dot(H, np.array([keypoint2[m][0], keypoint2[m][1], 1]))
					newx = newPoint[0]/newPoint[2]
					newy = newPoint[1]/newPoint[2]
					# Calculating the sum squared difference between the warped point and the corresponding keypoint features form the lastImage
					ssd = (float(keypoint1[m][0]) - newx)**2 + (float(keypoint1[m][1]) - newy)**2
					# If the sum squared difference is less than 50
					if ssd < 50:
						# Then add those coordinates as inliers
						inliers = inliers + 1
						inliersList.append(m)

			# If number of inliers is greater then maxInliers then update maxInliers and its corresponding information
			if inliers > maxInliers:
				maxInliers = inliers
				maxList = inliersList
				maxH = H

			itr = itr + 1 # Updating iterations by 1

		print "Number of Inliers: " + str(inliers)
		print "Percent: " + str(percent)
		print "Number of Itr: " + str(itr)

		if inliers <= 5:
			print "Stitching not possible due to less Overlapping"
			break
		
		"""
		plotImage6 = np.hstack((lastImage, newImage))		
		for i in inliersList:
			cv.circle(plotImage6, (keypoint1[i][0], keypoint1[i][1]), 4, (255, 0, 0), 2) 
			cv.circle(plotImage6, (keypoint2[i][0]+lastGray.shape[1], keypoint2[i][1]), 4, (255, 0, 0), 2) 
			cv.line(plotImage6, (keypoint1[i][0], keypoint1[i][1]), (keypoint2[i][0]+lastGray.shape[1], keypoint2[i][1]), (0, 0, 255), 1)

		cv.imshow('RANSACMatching', plotImage6)
		cv.waitKey(0)
		cv.destroyAllWindows()
		"""
		"""
		Image Warping + Blending
		Save Panorama output as mypano.png
		"""
		# List containing three corners of the image(Top left, Top Right, Bottom Left)
		checkList = [[1, 1, 1], [newImage.shape[0], 1, 1], [1, newImage.shape[1], 1]]
		checkx = [] # List for storing warped x-coordinates of the image
		checky = [] # List for storing warped y-coordinates of the image
		for coord in checkList: # Looping over the corner points
			# Calculating warped points
			checkCoord = np.matmul(H, np.array(coord))
			checkCoordx = checkCoord[0]/checkCoord[2]
			checkCoordy = checkCoord[1]/checkCoord[2]
			checkx.append(int(checkCoordx))
			checky.append(int(checkCoordy))
		
		# Calculating translation matrix if applicable
		# Checking if x and y coordinate of the warped points are negative
		if min(checkx) < 0 and min(checky) < 0:
			trans = np.array([[1.0, 0.0, -(min(checkx)-10)*1.0], [0.0, 1.0, -(min(checky)-10)*1.0], [0.0, 0.0, 1.0]])
 		elif min(checky) < 0: # Checking if y coordinate of the warped points are negative
			trans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -(min(checky)-10)*1.0], [0.0, 0.0, 1.0]])
		elif min(checkx) < 0: # Checking if x coordinate of the warped points are negative
			trans = np.array([[1.0, 0.0, -(min(checkx)-10)*1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
		else: # Identity matrix if no translation is required 
			trans = np.identity(3)
	
		"""
		newList = [[1, 1, 1], [newImage.shape[1], 1, 1], [1, newImage.shape[0], 1], [newImage.shape[1], newImage.shape[0], 1]]
		newx = []
		newy = []
		for coord in newList:
			checkCoord = np.matmul(H, np.array(coord))
			checkCoordx = checkCoord[0]/checkCoord[2]
			checkCoordy = checkCoord[1]/checkCoord[2]
			newx.append(int(checkCoordx))
			newy.append(int(checkCoordy))
		print newx
		print newy
		H3, status = cv.findHomography(np.array([[newx[0], newy[0]], [newx[1], newy[1]], [newx[2], newy[2]], [newx[3], newy[3]]]), np.array([[newx[0], newy[0]], [newx[1], newy[1]], [newx[2], newy[2]], [newx[3]+0, newy[3]]])) 
		"""
		print trans
		print h1

		# Multiply current homography with previous homographies
		h1 = np.matmul(h1, H)
		# Multiply translation with the multiplication of homographies
		h1 = np.matmul(trans, h1)
		#h1 = np.matmul(h1, H3)
		final = cv.warpPerspective(final, trans, (2000, 2000))
		# Warping lastImage just with translation if any
		finalImage1 = cv.warpPerspective(lastImage, np.matmul(trans, h2), (2000, 2000))
		# Warping newImage with translation + Multiplication of homographies
		finalImage2 = cv.warpPerspective(newImage, h1, (2000, 2000))
	
		# Loops for stitching lastImage with warped newImage
		for row in range(2000):
			for column in range(2000):
				if index == 0:
					final[row, column, :] = finalImage1[row, column, :]
				if finalImage2[row, column, 0] != 0 and finalImage2[row, column, 1] != 0 and finalImage2[row, column, 2] != 0:
					final[row, column, :] = finalImage2[row, column, :]
		
		cv.imshow('Wraped', final)
		cv.imshow('2', finalImage2)
		cv.waitKey(0)
		cv.destroyAllWindows()

	cv.imwrite("Result.png", final)

    
if __name__ == '__main__':
    main()
