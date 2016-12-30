from __future__ import print_function;
from numpy import *
import numpy as np;
import scipy.io;
from numpy import random;
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
from scipy.io import loadmat, savemat
import cv2

import tensorflow as tf
import cPickle

# Modify the dimension to vectorize the data
train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 7))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''

    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

def create_new_input(input_img_matrix, datasize):
    '''This function takes in all images in a given data set with given size and n is used to indicate the index of actor from the act list.
    '''
    x_dummy = (np.random.random((datasize,)+ xdim)/255.).astype(float32)
    i = x_dummy.copy()
    for j in range(datasize): # This loop vectorizes the data
        i[j] = input_img_matrix[j].reshape((227, 227, 3))
        #i[j,:,:,:] = input_img_matrix[j, :, :, :]
    i = i-mean(i)
    net_data = load("bvlc_alexnet.npy").item()

    x = tf.Variable(i)

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1],     strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o,    s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1],     strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o,    s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h,  s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    conv4flatten = tf.reshape(conv4, [datasize,            int(prod(conv4.get_shape()[1:]))])
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    # The vectorized data needs to be flattened and stored in a numpy array
    new_input = conv4flatten.eval(session=sess)
    return new_input

def create_test_M():
    ''' This function creates a .mat file which stores all test images
    Input: xs
    Output: .mat file of TestSet for later processing
    where the data in the output file is in matrix forms, each matrix is nx1024 where n is the number of test examples for different emotions
    '''
    mdict = {}
    i = 0
    for i in range(xs.shape[0]): # This loop separates all emotions in test set
        img_arr = xs[i, :].reshape(32, 32)
        resized_img = imresize(img_arr, (227, 227))
        resized_img.resize((227, 227, 1))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB) # Convert image to RGB
        rgb_img = rgb_img.reshape((1, 227*227*3))
        if testset_name in mdict.keys():
            mdict[testset_name] = vstack((mdict[testset_name], rgb_img))
        else:
            mdict[testset_name] = rgb_img
    savemat('227'+testset_name+'.mat', mdict)   

def create_new_inputs():
    output = {}
    test_matrix = create_new_input(M[testset_name], M[testset_name].shape[0])
    output[testset_name] = test_matrix
    savemat('new_'+testset_name+'.mat', output)

## Load data
testset_name = 'public_test_images'
#testset_name = 'hidden_test_images'
data = loadmat(testset_name+'.mat');
xs = data[testset_name].T;
del data;
# Preprocess images
xs = xs.reshape((-1, 1024)).astype('float');
xs -= np.mean(xs, axis=1)[:, np.newaxis];
xs /= np.sqrt(np.var(xs, axis=1) + 0.01)[:, np.newaxis];
# Targets
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

# Create a .mat file for public test sets
create_test_M() # Note please run this function first, then comment this line and uncomment the lines below to avoid possible laggings.

# M = loadmat('227'+testset_name+'.mat')
# create_new_inputs()