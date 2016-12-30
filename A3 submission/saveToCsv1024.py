import numpy as np
import csv
import scipy.io;
import itertools;
from numpy import random;
from pylab import *
from scipy.io import loadmat, savemat

import tensorflow as tf
import cPickle

M_test = loadmat('public_test_images.mat')
reshapedTestImg = M_test['public_test_images'].reshape((418, 1024))
labels = loadmat('val_labels.mat')
x = tf.placeholder(tf.float32, [None, 1024])
nhid = 800
snapshot = cPickle.load(open("NN.pkl"))
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

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

output = {}
for i in range(1, reshapedTestImg.shape[0]+1):
    test_image = reshapedTestImg[i-1, :]
    test_image = test_image.reshape((1, 1024))
    output[i] = argmax(sess.run(y, feed_dict={x:test_image}))+1

hit = 0
for i in range(1, 419):
    if output[i] == labels['val_labels'][i-1]:
        hit += 1
print hit/418.0

# Set rest of the entries to 0
for i in range(419, 1254):
    output[i] = 0

# save to test.mat
testmat = {}
for key, value in output.iteritems():
    temp = []
    temp.append(value)
    testmat[str(key)] = temp[0]
savemat('predictions1024.mat', testmat)

# Write to the csv file
with open('predictions1024.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(output.items())