import numpy as np
import csv
import scipy.io;
from numpy import random;
from pylab import *
from scipy.io import loadmat, savemat

import tensorflow as tf
import cPickle

'''This code creates the predictions for both public and hidden test set.'''
M_test = loadmat('new_public_test_images.mat')
M_test_hidden = loadmat('new_hidden_test_images.mat')

# Prepare the variables to be used
x = tf.placeholder(tf.float32, [None, 64896])
nhid = 300
snapshot = cPickle.load(open("AlexNetNN.pkl"))
init_W0 = snapshot["W0"]
init_b0 = snapshot["b0"]
init_W1 = snapshot["W1"]
init_b1 = snapshot["b1"]

W0 = tf.Variable(init_W0)
b0 = tf.Variable(init_b0)

W1 = tf.Variable(init_W1)
b1 = tf.Variable(init_b1)

layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1

y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 7])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Write in the results to dictionary
output = {}
for i in range(1, M_test['public_test_images'].shape[0]+1):
    test_image = M_test['public_test_images'][i-1, :]
    test_image = test_image.reshape((1, 64896))
    output[i] = argmax(sess.run(y, feed_dict={x:test_image}))+1

'''To create predictions only for public test set, please UNCOMMENT the for-loop below'''
# Set rest of the entries to 0
# for i in range(419, 1254):
#     output[i] = 0

'''To create predictions only for public test test, please COMMENT the for-loop below'''
# Set rest of the entries as predictions to hidden test set
for i in range(419, 1254):
    j = i-419
    test_image = M_test_hidden['hidden_test_images'][j, :]
    test_image = test_image.reshape((1, 64896))
    output[i] = argmax(sess.run(y, feed_dict={x:test_image}))+1
    
# save to test.mat
testmat = {}
for key, value in output.iteritems():
    temp = []
    temp.append(value)
    testmat[str(key)] = temp[0]
savemat('predictions.mat', testmat)

'''Note in python, entries of dictionary are sorted by keys automatically, so please add the header entry(Id, Prediction) manually.'''
# Write to the csv file
with open('predictions.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(output.items())