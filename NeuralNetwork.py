# -*- coding: utf-8 -*-
"""
@author: sahit
"""
import cv2
import os
# two dictionaries that map integers to images, i.e.,
# 2D numpy array.
TRAIN_IMAGE_DATA = {}
TEST_IMAGE_DATA  = {}
# the train target is an array of 1's
TRAIN_TARGET = []
# the set target is an array of 0's.
TEST_TARGET  = []
### Global counters for train and test samples
NUM_TRAIN_SAMPLES = 0
NUM_TEST_SAMPLES  = 0
## define the root directory
ROOT_DIR = 'C:/Users/sahit/Desktop/nn_train/'
## read the single bee train images
YES_BEE_TRAIN = ROOT_DIR + 'single_bee_train'
for root, dirs, files in os.walk(YES_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
            TRAIN_TARGET.append(int(1))
            NUM_TRAIN_SAMPLES +=1
            
 ## read the single bee test images
YES_BEE_TEST = ROOT_DIR + 'single_bee_test'
for root, dirs, files in os.walk(YES_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TEST_IMAGE_DATA[NUM_TEST_SAMPLES] = img
            TEST_TARGET.append(int(1))
            NUM_TEST_SAMPLES += 1
        
  ## read the no-bee train images
NO_BEE_TRAIN = ROOT_DIR + 'no_bee_train'
for root, dirs, files in os.walk(NO_BEE_TRAIN):
    for item in files:
          if item.endswith('.png'):
              ip = os.path.join(root, item)
              img = (cv2.imread(ip)/float(255))
              TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
              TRAIN_TARGET.append(int(0))
              NUM_TRAIN_SAMPLES += 1
# read the no-bee test images
NO_BEE_TEST = ROOT_DIR + 'no_bee_test'
for root, dirs, files in os.walk(NO_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TEST_IMAGE_DATA[NUM_TEST_SAMPLES] = img
            TEST_TARGET.append(int(0))
            NUM_TEST_SAMPLES += 1
print (NUM_TRAIN_SAMPLES)
print (NUM_TEST_SAMPLES)

train_images = TRAIN_IMAGE_DATA.values()
test_images = TEST_IMAGE_DATA.values()

import numpy as np

image_train = np.array(list(train_images))
image_test = np.array(list(test_images))
image_train_label = np.array(TRAIN_TARGET)
image_test_label = np.array(TEST_TARGET)


import keras
image_train_label = keras.utils.to_categorical(image_train_label,2)
image_test_label = keras.utils.to_categorical(image_test_label,2)


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

import tensorflow as tf
sess = tf.InteractiveSession()

#creating the model 
x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 32, 32, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8*8*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Cross entropy gives a measure of how far off the prediction is from the ground truth.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
#test the trained model
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
#save the model
saver = tf.train.Saver()
with tf.device("/gpu:0"):
    
    for i in range(80000):
        batch = next_batch(50,image_train,image_train_label)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("iteration %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy is %g"%accuracy.eval(feed_dict={x: image_test, y_: image_test_label, keep_prob: 1.0}))
save_path = saver.save(sess, "C:/Users/sahit/Desktop/nn_train/MODEL.ckpt")
print("model is saved in file: %s" %save_path)

#restore the model
# Running a new session
print("Starting 2nd session...")
with tf.Session() as sess:
    # Restore model weights from previously saved model
    save_restore = saver.restore(sess, "C:/Users/sahit/Desktop/nn_train/MODEL.ckpt")
    print("Model restored from file: %s" % save_path)

#defined netpath where the persisted NN is saved
netpath=save_path
def testNet(netpath, dirpath):

    




