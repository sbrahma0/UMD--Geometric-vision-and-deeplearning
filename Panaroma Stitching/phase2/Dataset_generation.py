#!/usr/bin/env python

import numpy as np
import cv2 as cv
import argparse
import glob
import random
import  os


def main():
	'''
	Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	
	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	Parse Command Line arguments
	'''
	
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath',
                        default='../Data/Train/',
                        help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/CIFAR10')
	Parser.add_argument('--SavePatchPath',
                        default='../Data/TrainPatch/',
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--SaveGtPath',
                        default='../Data/TrainGroundTruth/',
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--SaveHPath',
                        default='TxtFiles/TrainLabels.txt',
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
	Args = Parser.parse_args()

	BasePath = Args.BasePath  # Read path images
	SavePatchPath = Args.SavePatchPath  # Save the stitched or any images related to a
	SaveGtPath = Args.SaveGtPath
	SaveHPath = Args.SaveHPath
	images_col = []

	"""
	Read a set of images for Panorama stitching
	"""
	a = 1
	file = open(SaveHPath,'w')
	for img in glob.glob(BasePath + '/*.*'):
		
		g1 = cv.imread(img)
		h,w,_ = g1.shape
		ran = np.arange(-20,20)
		pts = random.sample(list(ran),8)

		patch = np.asarray(g1[int((h/2)-64):int((h/2)+64),int((w/2)-64):int((w/2) + 64)])
		cv.imwrite(SaveGtPath+str(a)+'.png', patch)
		top_left = [int((w/2)-64), int((h/2)-64)]
		bot_left = [int((w/2)-64), int((h/2)+64)]
		bot_right = [int((w/2)+64), int((h/2) + 64)]
		top_right = [int((w/2)+64), int((h/2) - 64)]

		# perturbation
		top_left_p = [int((w / 2) - 64+pts[0]), int((h / 2) - 64+pts[1])]
		bot_left_p = [int((w / 2) - 64+pts[2]), int((h / 2) + 64+pts[3])]
		bot_right_p = [int((w / 2) + 64+pts[4]), int((h / 2) + 64+pts[5])]
		top_right_p = [int((w / 2) + 64+pts[6]), int((h / 2) - 64+pts[7])]

		# h4t
		h4t = [[top_left[0]-top_left_p[0],top_left[1]-top_left_p[1]],[top_right[0]-top_right_p[0],top_right[1]-top_right_p[1]],
               [bot_right[0]-bot_right_p[0],bot_right[1]-bot_right_p[1]],[bot_left[0]-bot_left_p[0],bot_left[1]-bot_left_p[1]]]

		file.write(str(a)+' '+str(h4t[0][0])+' '+str(h4t[1][0])+' '+str(h4t[2][0])+' '+str(h4t[3][0])+' '+str(h4t[0][1])+' '
                   +str(h4t[1][1])+' '+str(h4t[2][1])+' '+str(h4t[3][1])+' '+str(top_left[0])+' '+str(top_left[1])+'\n')

		pts1 = [top_left,top_right,bot_right,bot_left]
		pts2 = [top_left_p, top_right_p, bot_right_p, bot_left_p]
		h_matrix, status = cv.findHomography(np.asarray(pts1), np.asarray(pts2))

		new = cv.warpPerspective(g1,np.linalg.inv(h_matrix),(w,h))

		cv.rectangle(g1, (top_left[0],top_left[1]), (bot_right[0],bot_right[1]), [255], 2)

		pp1 = np.dot(np.linalg.inv(h_matrix), [top_left_p[0], top_left_p[1], 1])
		pp2 = np.dot(np.linalg.inv(h_matrix), [top_right_p[0], top_right_p[1], 1])
		pp3 = np.dot(np.linalg.inv(h_matrix), [bot_right_p[0], bot_right_p[1], 1])
		pp4 = np.dot(np.linalg.inv(h_matrix), [bot_left_p[0], bot_left_p[1], 1])

		patch2 = np.asarray(new[int(pp1[1] / pp1[2]):int(pp1[1] / pp1[2])+128,int(pp1[0] / pp1[2]):int(pp1[0] / pp1[2])+128])

		print patch2.shape
		cv.imwrite(SavePatchPath+str(a)+'.png',patch2)

		"""
		cv.imshow('a',g1)
		cv.imshow('2', new)
		cv.imshow('fff', patch)
		cv.imshow('s',patch2)
		cv.waitKey(0)
		cv.destroyAllWindows()
		"""

		print str(a)+'Saved'
		a = a+1
		file.close

if __name__ == '__main__':
	main()
