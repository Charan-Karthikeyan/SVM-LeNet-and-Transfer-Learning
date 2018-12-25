#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow import keras
import random
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import os 
cwd = os.getcwd()


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", reshape=False)

train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labesl = mnist.test.labels
va_img = mnist.validation.images
va_label = mnist.validation.labels
train_images = np.pad(train_images,((0,0),(2,2),(2,2),(0,0)),'constant')
test_images = np.pad(test_images,((0,0),(2,2),(2,2),(0,0)),'constant')
va_img = np.pad(va_img,((0,0),(2,2),(2,2),(0,0)),'constant')
#print(train_images.shape,train_labels.shape)


# In[3]:


epoch = 10
batch_size = 109

def lecnn(x):
    mu = 0
    sigma = 0.1
    
    conv1 = tf.nn.conv2d(x,tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean = mu, stddev = sigma)),strides=[1,1,1,1],padding='VALID')
    print(conv1.shape)
    conv1 = conv1+tf.Variable(tf.zeros(6))
    print(conv1.shape)
    conv1 = tf.nn.relu(conv1)
    print(conv1.shape)
    pool1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1],strides=[1,2,2,1],padding='VALID')
    print(pool1.shape)
    conv2 = tf.nn.conv2d(pool1,tf.Variable(tf.truncated_normal(shape=[5,5,6,16],mean =mu,stddev=sigma)),strides=[1,1,1,1],padding="VALID")+tf.Variable(tf.zeros(16))
    conv2 = tf.nn.relu(conv2)
    
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    #Flatten the layer
    flat1 = flatten(pool2)
    print(flat1.shape)
    #fully connencted layer
    fc1 = tf.matmul(flat1,tf.Variable(tf.truncated_normal(shape=(400,120),mean=mu,stddev=sigma)))+tf.Variable(tf.zeros(120))
    print(fc1.shape)
    fc2 = tf.matmul(fc1,tf.Variable(tf.truncated_normal(shape=(120,84),mean=mu,stddev=sigma)))+tf.Variable(tf.zeros(84))
    #print(fc1.shape)
    
    #Logits layer 
    logits = tf.matmul(fc2,tf.Variable(tf.truncated_normal(shape=(84,10),mean=mu,stddev=sigma)))+tf.Variable(tf.zeros(10))
    return logits


# In[4]:


x_x = tf.placeholder(tf.float32,(None,32,32,1))
y = tf.placeholder(tf.int32,(None))
#one_hot_y = tf.one_hot(y,10)

#Training 
logits = lecnn(x_x)
cross_entrop = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y,10),logits=logits)
loss_op  =tf.reduce_mean(cross_entrop)
opti = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = opti.minimize(loss_op)


# In[6]:


#Model Evaluation
correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(tf.one_hot(y,10),1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
saver = tf.train.Saver()

def eval(image_data,label_data):
    num_examples = len(image_data)
    total_acc = 0
    sess = tf.get_default_session()
    for offset in range(0,num_examples,batch_size):
        x_data,y_data = image_data[offset:offset+batch_size],label_data[offset:offset+batch_size]
        accu = sess.run(accuracy,feed_dict={x_x:x_data,y:y_data})
        total_acc += (accu*len(x_data))
    return total_acc/num_examples

#Model Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    no_eg = len(train_images)
    
    print("Training the network")
    for i in range(epoch):
        train_images,train_labels = shuffle(train_images,train_labels)
        for offset in range (0,no_eg,batch_size):
            end = offset+batch_size
            x_data,y_data = train_images[offset:end],train_labels[offset:end]
            sess.run(training_op,feed_dict={x_x:x_data,y:y_data})
        
        validation_acc = eval(va_img,va_label)
        print("Epoch no:",i+1)
        print("The Validation Accuracy:",validation_acc)
        
    saver.save(sess,cwd+'/lenet')# change directory when running

        


# In[7]:


#Evalution of the Trained Model
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('.'))
    test_acc = eval(test_images,test_labesl)
    print("The test accuracy is",test_acc)

