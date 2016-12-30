from __future__ import print_function;
import numpy as np;
import scipy.io;
import itertools;
from numpy import random;
from sklearn import neighbors;
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

def create_training_M():
    ''' This function creates a .mat file which stores all training images
    Input: xs, labels 
    Output: 'training.mat' for later processing
    where the data in the output file is in matrix forms as 'train1' .... 'train7', each matrix is nx1024 where n is the number of training examples for different emotions
    '''
    mdict = {}
    i = 0
    for label in labels: # This loop separates all emotions in different training sets
        if 'train'+str(label) in mdict.keys():
            mdict['train'+str(label)] = vstack((xs[i, :], mdict['train'+str(label)]))
        else:
            mdict['train'+str(label)] = xs[i, :]
        i += 1 
    savemat('NNtraining.mat', mdict)

## Load data
data = scipy.io.loadmat('labeled_images.mat');
ids = data['tr_identity']
labels = data['tr_labels'][:, 0];
xs = data['tr_images'].T;
del data;
# Preprocess images

xs = xs.reshape((-1, 1024)).astype('float');
xs -= np.mean(xs, axis=1)[:, np.newaxis];
xs /= np.sqrt(np.var(xs, axis=1) + 0.01)[:, np.newaxis];
# Make ids unique (for -1s)
inc = itertools.count(start=-1, step=-1).__iter__();
uids = [id[0] if id != -1 else inc.next() for id in ids];
# Targets
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

# Create a .mat file for training sets
create_training_M()
########################################################################################################################
M = loadmat("NNtraining.mat")
def get_train_batch(M, N):
    n = N/10
    batch_xs = zeros((0, 1024))
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
    batch_xs = zeros((0, 1024))
    batch_y_s = zeros( (0, 7))
    
    train_k =  ["train"+str(i) for i in range(1, 8)]
    for k in range(7):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(7)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s


x = tf.placeholder(tf.float32, [None, 1024])


nhid = 800
# W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
# b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
# 
# W1 = tf.Variable(tf.random_normal([nhid, 7], stddev=0.01))
# b1 = tf.Variable(tf.random_normal([7], stddev=0.01))

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

lam = 0.0001
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
NLL = -tf.reduce_sum(y_*tf.log(y)+decay_penalty)

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

trainCR = array([])
h = array([])
for i in range(10000):
  #print i  
  batch_xs, batch_ys = get_train_batch(M, 50)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 1 == 0:
    print ("i=",i)

    train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xs, batch_ys = get_train(M)


    trainCR = np.append(trainCR, train_accuracy)

    h = np.append(h, i)

    # Modify nhid to get 300 and 800 results
    if i == 9999:
        snapshot = {}
        snapshot["W0"] = sess.run(W0)
        snapshot["W1"] = sess.run(W1)
        snapshot["b0"] = sess.run(b0)
        snapshot["b1"] = sess.run(b1)
        cPickle.dump(snapshot,  open('NN'+'.pkl', "w"))

plt.plot(h, trainCR, 'r', label = "training set")
plt.title('Correct classification rate vs Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Correct classification rate')
plt.legend(loc='lower right')
plt.show()