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
from tensorflow.keras import layers
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    
    with tf.compat.v1.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases
        
        return layer, weights

def new_pool_layer(input, name):
    
    with tf.compat.v1.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool2d(input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        return layer

def new_relu_layer(input, name):
    
    with tf.compat.v1.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        
        return layer

def new_fc_layer(input, num_inputs, num_outputs, name):
    
    with tf.compat.v1.variable_scope(name) as scope:

        # Create new weights and biases.
        weights = tf.Variable(tf.random.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
        
        return layer

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
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
	training = tf.compat.v1.placeholder_with_default(False, shape=(), name='training')

	# Convolutional Layer 1
	layer_conv1, weights_conv1 = new_conv_layer(input=Img, num_input_channels=3, filter_size=5, num_filters=20, name ="conv1")

	# RelU layer 1
	layer_relu1 = new_relu_layer(layer_conv1, name="relu1")

	#Batch_norm 1
	batch_norm1 = tf.layers.batch_normalization(layer_relu1, training=training, momentum=0.9)

	# Convolutional Layer 2
	layer_conv2, weights_conv2 = new_conv_layer(input=batch_norm1, num_input_channels=20, filter_size=3, num_filters=40, name= "conv2")

	# RelU layer 2
	layer_relu2 = new_relu_layer(layer_conv2, name="relu2")

	#Batch_norm 2
	batch_norm2 = tf.layers.batch_normalization(layer_relu2, training=training, momentum=0.9)

	# Convolutional Layer 3
	layer_conv3, weights_conv3 = new_conv_layer(input=batch_norm2, num_input_channels=40, filter_size=3, num_filters=80, name= "conv3")

	# RelU layer 3
	layer_relu3 = new_relu_layer(layer_conv3, name="relu3")

	#Batch_norm 3
	batch_norm3 = tf.layers.batch_normalization(layer_relu3, training=training, momentum=0.9)

	# Pooling Layer 1
	layer_pool1 = new_pool_layer(batch_norm3, name="pool1")

	# Flatten Layer
	num_features = layer_pool1.get_shape()[1:4].num_elements()
	layer_flat = tf.reshape(layer_pool1, [-1, num_features])

	# Fully-Connected Layer 1
	layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=10, name="fc1")

	# Use Softmax function to normalize the output
	with tf.compat.v1.variable_scope("Softmax"):
		y_pred = tf.nn.softmax(layer_fc1)
		y_pred_cls = tf.argmax(y_pred, axis=1)

	prLogits = layer_fc1
	prSoftMax = y_pred
	
	return prLogits, prSoftMax

def ResNetModel(Img, ImageSize, MiniBatchSize):
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
	training = tf.compat.v1.placeholder_with_default(False, shape=(), name='training')

	#Batch_norm 1
	batch_norm1 = tf.layers.batch_normalization(Img, training=training, momentum=0.9)

	# RelU layer 1
	layer_relu1 = new_relu_layer(batch_norm1, name="relu1")
	
	# Convolutional Layer 1
	layer_conv1, weights_conv1 = new_conv_layer(layer_relu1, num_input_channels=3, filter_size=5, num_filters=20, name ="conv1")

	#Batch_norm 2
	batch_norm2 = tf.layers.batch_normalization(layer_conv1, training=training, momentum=0.9)

	# RelU layer 2
	layer_relu2 = new_relu_layer(batch_norm2, name="relu2")

	# Convolutional Layer 2
	layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu2, num_input_channels=20, filter_size=3, num_filters=40, name= "conv2")

	#Skip-Connection 1
	#paddings = tf.constant([[0, 0], [8, 8], [8, 8], [0, 0]])
	#batch_norm1pad = tf.pad(batch_norm1, paddings, "CONSTANT")
	unit_conv1, weights_unit_conv1 = new_conv_layer(input=Img, num_input_channels=3, filter_size=1, num_filters=40, name ="unitconv1")
	skip_conn1 = layers.Add()([unit_conv1, layer_conv2])

	#Batch_norm 3
	batch_norm3 = tf.layers.batch_normalization(skip_conn1, training=training, momentum=0.9)

	# RelU layer 3
	layer_relu3 = new_relu_layer(batch_norm3, name="relu3")

	# Convolutional Layer 3
	layer_conv3, weights_conv3 = new_conv_layer(input=layer_relu3, num_input_channels=40, filter_size=3, num_filters=80, name= "conv3")

	#Batch_norm 4
	batch_norm4 = tf.layers.batch_normalization(layer_conv3, training=training, momentum=0.9)

	# RelU layer 4
	layer_relu4 = new_relu_layer(batch_norm4, name="relu3")

	# Convolutional Layer 4
	layer_conv4, weights_conv4 = new_conv_layer(input=layer_relu4, num_input_channels=80, filter_size=3, num_filters=100, name= "conv4")

	#Skip-Connection 2
	unit_conv2, weights_unit_conv2 = new_conv_layer(input=skip_conn1, num_input_channels=40, filter_size=1, num_filters=100, name ="unitconv2")
	skip_conn2 = layers.Add()([unit_conv2, layer_conv4])

	#Batch_norm 5
	batch_norm5 = tf.layers.batch_normalization(skip_conn2, training=training, momentum=0.9)

	# RelU layer 5
	layer_relu5 = new_relu_layer(batch_norm5, name="relu5")

	# Pooling Layer 1
	layer_pool1 = new_pool_layer(layer_relu5, name="pool1")

	# Flatten Layer
	num_features = layer_pool1.get_shape()[1:4].num_elements()
	layer_flat = tf.reshape(layer_pool1, [-1, num_features])

	# Fully-Connected Layer 1
	layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=10, name="fc1")

	# Use Softmax function to normalize the output
	with tf.compat.v1.variable_scope("Softmax"):
		y_pred = tf.nn.softmax(layer_fc1)
		y_pred_cls = tf.argmax(y_pred, axis=1)

	prLogits = layer_fc1
	prSoftMax = y_pred
	
	return prLogits, prSoftMax

def name_block(s,b):
    """
    s is the integer for the number of each set of residual blocks
    """
    s = 'set'+str(b)+'block'+str(s)
    return s

def concatenation(out):
    return tf.concat(out, axis = 3)

def ideal_block(Img, num_filters1, num_filters2, kernel_size1, kernel_size2,s,b):
    net = Img
    name = name_block(s=s,b=b)
    # net = tf.layers.conv2d(inputs = net, name=name+'layer_res_conv_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    # net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'layer_bn0')
    # I_store = net

    net = tf.layers.conv2d(inputs = net, name=name +'conv_1', padding='same',filters = num_filters1, kernel_size = kernel_size1, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'layer_bn1')
    net = tf.nn.relu(net, name = name +'layer_Relu1')

    #Define 2nd Layer of the convolution
    net = tf.layers.conv2d(inputs = net, name=name+'conv_2', padding='same',filters = num_filters2, kernel_size = kernel_size2, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'layer_bn2')
    return net

def split(Img, cardinality, num_filters1, num_filters2, kernel_size1, kernel_size2, b):
    """
    Img is the input from each path taken by the input
    cardinality is the number of paths to be taken
    """
    # out = ideal_block(Img = Img, num_filters1 = num_filters1, num_filters2 = num_filters2, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, s = 0, b=b)
    out = list()
    for i in range(cardinality):
        # net = Img
        net = ideal_block(Img = Img, num_filters1 = num_filters1, num_filters2 = num_filters2, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, s = i, b=b)
        out.append(net)

    return concatenation(out)

def merge_block(Img, cardinality,num_filters, num_filters1, num_filters2,kernel_size, kernel_size1, kernel_size2, b):
    net = Img
    name = name_block(s=b,b=b)
    net = tf.layers.conv2d(inputs = net, name=name+'merge_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'merge_bn0')
    I_store = net

    net = split(Img = net, cardinality = cardinality, num_filters1 = num_filters1, num_filters2 = num_filters2, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, b = b)
    net = tf.layers.conv2d(inputs = net, name=name+'split_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'split_n')

    out = tf.math.add(net, I_store)

    net = tf.nn.relu(out, name = name +'layer_Relu2')

    return net



def ResNext(Img, ImageSize, MiniBatchSize):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    #Define Filter parameters for the first convolution layer block
    filter_size1 = 1
    num_filters1 = 16
    n1 = 1                  #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size2 = 5
    num_filters2 = 32
    n2 = 1                #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size = 3
    num_filters = 16
    n = 1                  #number of residual blocks in each convolution layer blocks


    #Define number of class labels
    num_classes = 10
    #############################
    # Fill your network here!
    #############################

    #Define net placeholder
    net = Img
    #Construct first convolution block of n residual blocks
    net = merge_block(Img = net, cardinality = 5,num_filters = num_filters, num_filters1 = num_filters1, num_filters2 = num_filters2,kernel_size = filter_size, kernel_size1= filter_size1, kernel_size2= filter_size2, b = 1)
    # net = merge_block(Img = net, cardinality = 3,num_filters = num_filters, num_filters1 = num_filters1, num_filters2 = 16,kernel_size = filter_size, kernel_size1= filter_size1, kernel_size2= filter_size2, b = 2)

    # net = n_res_block(net, num_filters = num_filters4, kernel_size = filter_size4, n_blocks = n4, b=4, downsampling =True)

    #Define flatten_layer
    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

    net = tf.layers.dense(inputs = net, name ='layer_fc2',units=256, activation=tf.nn.relu)

    # net = tf.layers.dense(inputs = net, name ='layer_fc3',units=64, activation=tf.nn.relu)

    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = num_classes, activation = None)


    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax

def denseBlock(Img, num_layers, len_dense,num_filters,kernel_size,downsampling = False):
    with tf.variable_scope("dense_unit"+str(num_layers)):
        nodes = []
        # img = tf.layers.conv2d(inputs = Img,padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = None)
        img = tf.layers.conv2d(inputs = Img, padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)

        nodes.append(img)
        for z in range(len_dense):
            img = tf.nn.relu(Img)
            img = tf.layers.conv2d(inputs = img, padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
            net = tf.layers.conv2d(inputs = concatenation(nodes), padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
            nodes.append(net)
        return net
    # net = tf.layers.conv2d(inputs = net, name=name+'split_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    # net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'split_n')

def DenseNet(Img, ImageSize, MiniBatchSize):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    #Define Filter parameters for the first convolution layer block
    filter_size1 = 5
    num_filters1 = 20
    n1 = 1                  #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size2 = 3
    num_filters2 = 40
    n2 = 2                 #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size = 3
    num_filters = 80
    n = 1                  #number of residual blocks in each convolution layer blocks


    #Define number of class labels
    num_classes = 10
    #############################
    # Fill your network here!
    #############################

    #Define net placeholder
    net = Img

    net = tf.layers.conv2d(net,num_filters1,kernel_size = filter_size1,activation = None)

    #Construct first convolution block of n residual blocks
    net  = denseBlock(Img = net, num_layers = 1, len_dense = 4,num_filters = num_filters1,kernel_size =filter_size1 ,downsampling = False)
    # net = n_res_block(net, num_filters = num_filters4, kernel_size = filter_size4, n_blocks = n4, b=4, downsampling =True)

    #Define flatten_layer
    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    #net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

    #net = tf.layers.dense(inputs = net, name ='layer_fc2',units=256, activation=tf.nn.relu)

    # net = tf.layers.dense(inputs = net, name ='layer_fc3',units=64, activation=tf.nn.relu)

    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = num_classes, activation = None)


    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax
