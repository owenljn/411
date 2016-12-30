from __future__ import print_function;
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

import tensorflow as tf
import cPickle

temp = loadmat("newTraining.mat")
M = {}
M_valid = {}
# Process the input and use part of the training set as validation set

for i in range(1, 8):
    M['train'+str(i)] = temp['train'+str(i)]
    M_valid['valid'+str(i)] = temp['train'+str(i)][270:, :]
del temp

def get_train_batch(M, N):
    ''' This function uses mini-batch linear regression to select a mini-batch size of N out of input M.
    Input: M, N
    Output: batch_xs, batch_y_s
    '''
    n = N/10
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 7))
    
    train_k =  ["train"+str(i) for i in range(1, 8)]

    train_size = len(M[train_k[0]])
    
    for k in range(7):
        train_size = len(M[train_k[k]])
        idx = array(np.random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(7)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s

def get_train(M):
    ''' This function returns a stacked images matrix and its label matrix from training set M for later processing.
    Input: M
    Output: batch_xs, batch_y_s
    '''
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 7))
    
    train_k =  ["train"+str(i) for i in range(1, 8)]
    for k in range(7):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(7)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_valid(M):
    ''' This function returns a stacked images matrix and its label matrix from validation set M for later processing.
    Input: M
    Output: batch_xs, batch_y_s
    '''
    batch_xs = zeros((0, 64896))
    batch_y_s = zeros( (0, 7))
    
    valid_k =  ["valid"+str(i) for i in range(1, 8)]
    for k in range(7):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = zeros(7)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    

x = tf.placeholder(tf.float32, [None, 64896])


nhid = 300

# To train the network from the begging, uncomment the code below.
# W0 = tf.Variable(tf.random_normal([64896, nhid], stddev=0.01))
# b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
# 
# W1 = tf.Variable(tf.random_normal([nhid, 7], stddev=0.01))
# b1 = tf.Variable(tf.random_normal([7], stddev=0.01))

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

lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
valid_x, valid_y = get_valid(M_valid)

trainCR = array([])
validCR = array([])
testCR = array([])
h = array([])
for i in range(1500):
    # Mini-batch linear regression with a batch size of 50
    print ("i=",i)
    batch_xs, batch_ys = get_train_batch(M, 50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
    if i % 1 == 0:

        valid_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

        batch_xs, batch_ys = get_train(M)

        trainCR = np.append(trainCR, train_accuracy)
        validCR = np.append(validCR, valid_accuracy)
        h = np.append(h, i)

    # Save the trained weights and biases for part 3
    # Modify nhid to get 300 and 800 results
    if i == 1499:
        snapshot = {}
        snapshot["W0"] = sess.run(W0)
        snapshot["W1"] = sess.run(W1)
        snapshot["b0"] = sess.run(b0)
        snapshot["b1"] = sess.run(b1)
        cPickle.dump(snapshot,  open("AlexNetNN.pkl", "w"))

# Plot out the results
print ("The final performance classification on the training set is: ", train_accuracy)
print ("The final performance classification on the validation set is: ", valid_accuracy)
plt.plot(h, trainCR, 'r', label = "training set")
plt.plot(h, validCR, 'g', label = "validation set")
plt.title('Correct classification rate vs Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Correct classification rate')
plt.legend(loc='lower right')
plt.show()