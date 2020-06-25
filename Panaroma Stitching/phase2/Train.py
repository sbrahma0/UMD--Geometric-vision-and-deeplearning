#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import SupervisedModel, UnsupervisedModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(PatchPath, GtPath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(1, len(DirNamesTrain)-1)
        RandImageName1 = GtPath +'/'+ str(RandIdx) + '.png'
        RandImageName2 = PatchPath +'/'+ str(RandIdx) + '.png'
        ImageNum += 1
    	#print RandImageName1
    	##########################################################
    	# Add any standardization or data augmentation here!
    	##########################################################

        I1 = np.float32(cv2.imread(RandImageName1))
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I1 = iu.StandardizeInputs(I1)
        I2 = np.float32(cv2.imread(RandImageName2))
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        I2 = iu.StandardizeInputs(I2)

        imgData = np.dstack((I1,I2))
        var = TrainLabels
        line = var[RandIdx-1]       
        _,a,b,c,d,e,f,g,h,_,_ = line.split(' ')
        Label = [int(a)/20, int(b)/20, int(c)/20, int(d)/20, int(e)/20, int(f)/20, int(g)/20, int(h)/20]
        I1Batch.append(imgData)
        LabelBatch.append(Label)
        
    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, PatchPath, GtPath, LogsPath, ModelType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Predict output with forward pass
    if ModelType == 'Sup':
        prSoftMax = SupervisedModel(ImgPH, ImageSize, MiniBatchSize)
    elif ModelType == 'Unsup':
         prSoftMax = UnsupervisedModel(ImgPH, ImageSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        if ModelType == 'Sup':
            loss = tf.squeeze(tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.square(LabelPH - prSoftMax), axis=-1, keepdims=True)), axis = 0))
        elif ModelType == 'Unsup':
            loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.math.abs(ImgPH[:, :, :, 1::2] - prSoftMax), axis=1), axis=2), axis=1)
            loss = tf.squeeze(tf.reduce_mean(loss, axis=0))

    with tf.name_scope('Adam'):
    	###############################################
    	# Fill your optimizer of choice here!
    	###############################################
        Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.compat.v1.summary.scalar('LossEveryIter', loss)
	
    # tf.summary.image('Anything you want', AnyImg)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.compat.v1.summary.merge_all()

    # Setup Saver
    Saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session() as sess:    
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.compat.v1.summary.FileWriter(LogsPath, graph=tf.compat.v1.get_default_graph())
            
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                I1Batch, LabelBatch = GenerateBatch(PatchPath, GtPath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')

                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved...')
            

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--PatchPath', default='../Data/TrainPatch', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
    Parser.add_argument('--GtPath', default='../Data/TrainGroundTruth', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
    Parser.add_argument('--CheckPointPath', default='Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=10, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    PatchPath = Args.PatchPath
    GtPath = Args.GtPath    
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(GtPath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.compat.v1.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.compat.v1.placeholder(tf.float32, shape=(MiniBatchSize, 8))
    
    TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, PatchPath, GtPath, LogsPath, ModelType)
        
    
if __name__ == '__main__':
    main()
