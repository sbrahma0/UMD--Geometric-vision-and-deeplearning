"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
from Misc.Utils import *
from Misc.TFSpatialTransformer import *
# Don't generate pyc codes
sys.dont_write_bytecode = True

def SupervisedModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    # Convolutional Layers     
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(Img)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    #flattening layer
    x = tf.keras.layers.Flatten()(x)
    
    #Fully-connected layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5,noise_shape=None,seed=None)(x, training=True)
    H4pt = tf.keras.layers.Dense(8, activation=None)(x)

    return H4pt

def DLT(MiniBatchSize, h4pt):
    pts_1_tile = tf.expand_dims(h4pt, [2])
    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(h4pt, [2])
    # 4 points on the second image
    pred_pts_2_tile = tf.add(pred_h4p_tile, pts_1_tile)

    # Auxiliary tensors used to create Ax = b equation
    M1 = tf.constant(Aux_M1, tf.float32)
    M1_tensor = tf.expand_dims(M1, [0])
    M1_tile = tf.tile(M1_tensor,[MiniBatchSize,1,1])

    M2 = tf.constant(Aux_M2, tf.float32)
    M2_tensor = tf.expand_dims(M2, [0])
    M2_tile = tf.tile(M2_tensor,[MiniBatchSize,1,1])

    M3 = tf.constant(Aux_M3, tf.float32)
    M3_tensor = tf.expand_dims(M3, [0])
    M3_tile = tf.tile(M3_tensor,[MiniBatchSize,1,1])

    M4 = tf.constant(Aux_M4, tf.float32)
    M4_tensor = tf.expand_dims(M4, [0])
    M4_tile = tf.tile(M4_tensor,[MiniBatchSize,1,1])

    M5 = tf.constant(Aux_M5, tf.float32)
    M5_tensor = tf.expand_dims(M5, [0])
    M5_tile = tf.tile(M5_tensor,[MiniBatchSize,1,1])

    M6 = tf.constant(Aux_M6, tf.float32)
    M6_tensor = tf.expand_dims(M6, [0])
    M6_tile = tf.tile(M6_tensor,[MiniBatchSize,1,1])


    M71 = tf.constant(Aux_M71, tf.float32)
    M71_tensor = tf.expand_dims(M71, [0])
    M71_tile = tf.tile(M71_tensor,[MiniBatchSize,1,1])

    M72 = tf.constant(Aux_M72, tf.float32)
    M72_tensor = tf.expand_dims(M72, [0])
    M72_tile = tf.tile(M72_tensor,[MiniBatchSize,1,1])

    M8 = tf.constant(Aux_M8, tf.float32)
    M8_tensor = tf.expand_dims(M8, [0])
    M8_tile = tf.tile(M8_tensor,[MiniBatchSize,1,1])

    Mb = tf.constant(Aux_Mb, tf.float32)
    Mb_tensor = tf.expand_dims(Mb, [0])
    Mb_tile = tf.tile(Mb_tensor,[MiniBatchSize,1,1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
    A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
    A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
    A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

    A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                                   tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                                   tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
         tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8
    print('--Shape of A_mat:', A_mat.get_shape().as_list())
    # Form b matrix
    b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
    print('--shape of b:', b_mat.get_shape().as_list())

    # Solve the Ax = b
    H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
    print('--shape of H_8el', H_8el)

    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = tf.ones([MiniBatchSize, 1, 1])
    H_9el = tf.concat([H_8el,h_ones],1)
    H_flat = tf.reshape(H_9el, [-1,9])
    H_mat = tf.reshape(H_flat,[-1,3,3])

    return H_mat 

def transform(MiniBatchSize, h_mat, image):

    M = np.array([[128/2.0, 0., 128/2.0],
                  [0., 128/2.0, 128/2.0],
                  [0., 0., 1.]]).astype(np.float32)

    M_tensor  = tf.constant(M, tf.float32)
    M_tile   = tf.tile(tf.expand_dims(M_tensor, [0]), [MiniBatchSize, 1,1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv   = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [MiniBatchSize,1,1])
    # Transform H_mat since we scale image indices in transformer
    H_mat = tf.matmul(tf.matmul(M_tile_inv, h_mat), M_tile)
    # Transform image 1 (large image) to image 2
    out_size = (128, 128)
    warped_images, _ = transformer(image, H_mat, out_size)
    # warp image 2 to image 1
    
    # Extract the warped patch from warped_images by flatting the whole batch before using indices
    # Note that input I  is  3 channels so we reduce to gray
    warped_gray_images = tf.reduce_mean(warped_images, 3)
    warped_images_flat = tf.reshape(warped_gray_images, [-1])
    #patch_indices_flat = tf.reshape(self.patch_indices, [-1])
    #pixel_indices =  patch_indices_flat + self.batch_indices_tensor
    #pred_I2_flat = tf.gather(warped_images_flat, pixel_indices)

    pred_I2 = tf.reshape(warped_images_flat, [MiniBatchSize, 128, 128, 1])

    return pred_I2

def UnsupervisedModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    # Convolutional Layers     
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(Img)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=True)

    #flattening layer
    x = tf.keras.layers.Flatten()(x)
    
    #Fully-connected layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5,noise_shape=None,seed=None)(x, training=True)
    H4pt = tf.keras.layers.Dense(8, activation=None)(x)
    
    x = DLT(MiniBatchSize, H4pt)
    pred = transform(MiniBatchSize, x, Img[:, :, :, 0::2])

    return pred
