#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np 
import sklearn
import pandas as pd
from PIL import Image
import glob
from tqdm import tqdm
import os 
import cv2
import skimage
from sklearn.utils import class_weight 
from keras.utils.np_utils import to_categorical
from tensorflow.contrib.layers import flatten
#from random import shuffle
import random
import keras
from keras.applications.vgg19 import VGG19 as vgg
from keras.applications.vgg16 import VGG16 as vgg_1
from keras.models import Model
#from keras_applications.vgg19 as vgg
from keras.optimizers import SGD,RMSprop,Adam
from keras.models import load_model
import h5py
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D
from keras.layers import MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization
import cv2
from sklearn.utils import shuffle
import os 
cwd = os.getcwd()


# In[2]:


cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv(cwd+"/monkey_species/monkey_labels.txt",names = cols,skiprows =1)
labels = labels['Common Name']
train_img_dir = cwd+'/monkey_species/training/'
test_img_dir = cwd+'/monkey_species/validation/'


# In[3]:


def populate_image(folder):
    images = []
    labels = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['n0']:
                label = 0
            elif folderName in ['n1']:
                label = 1
            elif folderName in ['n2']:
                label = 2
            elif folderName in ['n3']:
                label = 3
            elif folderName in ['n4']:
                label = 4
            elif folderName in ['n5']:
                label = 5
            elif folderName in ['n6']:
                label = 6
            elif folderName in ['n7']:
                label = 7
            elif folderName in ['n8']:
                label = 8
            elif folderName in ['n9']:
                label = 9
            else:
                label = 10
            for image_filename in tqdm(os.listdir(folder+folderName)):
                img_file  = cv2.imread(folder+folderName+'/'+image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file,(150,150,3))
                    img_array = np.asarray(img_file)
                    images.append(img_array)
                    labels.append(label)
                    #print(labels)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels

#Populating the images and the labels 
train_images,train_labels = populate_image(train_img_dir)
test_images,test_labels = populate_image(test_img_dir)
#print(train_images.shape)


# In[4]:


#TODO change the params 
train_labels_hot = to_categorical(train_labels,num_classes = 10)
test_labels_hot = to_categorical(test_labels,num_classes=10)


# In[5]:


epoch  = 20
batch_size = 109
#x  = tf.placeholder(shape=(None,4,4,512),dtype='float32')
def network(x):
    mu = 0
    sigma = 0.1
    
    #input_layer = tf.nn.conv2d(x,tf.Variable(tf.truncated_normal(shape = [5,5,1,3],mean=mu,stddev=sigma)),strides=[1,1,1,1],padding="VALID")
    conv1 = tf.nn.conv2d(x,tf.Variable(tf.truncated_normal(shape =[1,1,3,64],mean = mu,stddev = sigma)),strides=[1,1,1,1],padding="VALID")+tf.Variable(tf.zeros(64))
    conv1 = tf.nn.relu(conv1)
    print("conv1",conv1.shape)
    
    #Pool of layer 1 with dimension (75,75,64)
    pool_1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    print("pool1",pool_1.shape)
    
    #convolution layer 2 with dimenion (75,75,128)
    conv2 = tf.nn.conv2d(pool_1,tf.Variable(tf.truncated_normal(shape = [1,1,64,128],mean=mu,stddev=sigma)),strides=[1,1,1,1],padding="VALID")+tf.Variable(tf.zeros(128))
    conv2 = tf.nn.relu(conv2)
    print("conv2",conv2.shape)
    
    #pool of layer 2 with dimenstion (37,37,128)
    pool_2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    print("Pool2",pool_2.shape)
    
    #convolution of layer 3 with dimenions (37,37,256)
    conv3 = tf.nn.conv2d(pool_2,tf.Variable(tf.truncated_normal(shape=[1,1,128,256],mean=mu,stddev=sigma)),strides=[1,1,1,1],padding="VALID")+tf.Variable(tf.zeros(256))
    conv3 = tf.nn.relu(conv3)
    print("conv3",conv3.shape)
    
    #pool of layer 3 with dimensions (18,18,256)
    pool_3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    print("pool3",pool_3.shape)
    
    #convolution of layer 3 with (18,18,512)
    conv4 = tf.nn.conv2d(pool_3,tf.Variable(tf.truncated_normal(shape=[1,1,256,512],mean=mu,stddev=sigma)),strides=[1,1,1,1],padding="VALID")+tf.Variable(tf.zeros(512))
    conv4 = tf.nn.relu(conv4)
    print("conv4",conv4.shape)
    
    #pool of layer of layer 4 with dimensions(9,9,512)
    pool_4 = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    print("pool4",pool_4.shape)
    
    #convolution layer 5 with dimensions(9,9,512)
    conv5 = tf.nn.conv2d(pool_4,tf.Variable(tf.truncated_normal(shape=[1,1,512,512],mean = mu,stddev=sigma)),strides=[1,1,1,1],padding="VALID")
    conv5 = tf.nn.relu(conv5)
    print("conv5",conv5.shape)
    
    #pool layer 5 with dimensions (4,4,512)
    pool_5 = tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    print("pool5",pool_5.shape)
    
    #Flatten the layer into (8192)
    flat1 = tf.layers.flatten(pool_5)
    print(flat1.shape)
    fc1 = tf.matmul(flat1,tf.Variable(tf.truncated_normal(shape=(8192,120),mean=mu,stddev=sigma)))+tf.Variable(tf.zeros(120))
    
    fc2 = tf.matmul(fc1,tf.Variable(tf.truncated_normal(shape=(120,84),mean=mu,stddev=sigma)))+tf.Variable(tf.zeros(84))
    
    #dense layer with (0,10)
    logits = logits = tf.matmul(fc2,tf.Variable(tf.truncated_normal(shape=(84,10),mean=mu,stddev=sigma)))+tf.Variable(tf.zeros(10))
    return logits


# In[6]:




# In[7]:


x = tf.placeholder(tf.float32,(None,150,150,3))
y = tf.placeholder(tf.int32,(None))
y_hot = tf.one_hot(y,10)
#Training 
logits = network(x)
#logits = lecnn(x)
cross_entrop = tf.nn.softmax_cross_entropy_with_logits(labels=y_hot,logits=logits)
loss_op = tf.reduce_mean(cross_entrop)
opti = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = opti.minimize(loss_op)


# In[8]:


correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(y_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
saver = tf.train.Saver()

def eval(image_data,label_data):
    num_examples = len(image_data)
    total_acc = 0
    sess = tf.get_default_session()
    for offset in range(0,num_examples,batch_size):
        x_data,y_data = image_data[offset:offset+batch_size],label_data[offset:offset+batch_size]
        accu = sess.run(accuracy,feed_dict ={x:x_data,y:y_data})
        total_acc += (accu*len(y_data))
    return total_acc/num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    no_eg = len(train_images)
    print("Training the Network")
    for i in range(epoch):
        train_images,train_labels = shuffle(train_images,train_labels)
        for offset in range (0,no_eg,batch_size):
            end = offset+batch_size
            x_data,y_data = train_images[offset:end],train_labels[offset:end]
            sess.run(training_op,feed_dict = {x:x_data,y:y_data})
        validation_acc = eval(test_images,test_labels)
        print("Epoch No:",i+1) 
        print("The accuracy is:",validation_acc*100) # accuracy is 63-66%


# In[9]:


class MetricsCheckpoint(Callback):
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


# In[13]:


# for the transfer learning algorithm
class_weights = class_weight.compute_class_weight("balanced",np.unique(train_labels),train_labels)
weights_dir = cwd +"/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
#load_model(weights_dir)
#TODO : Try changing the 
opti = keras.optimizers.Adam(lr = 0.001)
model_net = vgg_1(include_top = False,weights = weights_dir ,input_shape =(150,150,3))
def pre_net(train_img,train_lab,test_img,test_lab,model,weights,class_wei,num_class,num_epoch,optimizer):
    base_model = model
    
    x = base_model.output
    x = Flatten()(x)
    
    pred = Dense(num_class,activation='softmax')(x)
    model_1 = Model(inputs = base_model.input,output = pred)
    
    for layer in base_model.layers:
        layer.trainable = False
    model_1.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    callback_list = [keras.callbacks.EarlyStopping(monitor='validation_acc',patience=3,verbose=1)]
        
    model_fit = model_1.fit(train_img,train_lab,epochs=num_epoch,class_weight=class_wei,validation_data=(test_img,test_lab),verbose = 1,callbacks=[MetricsCheckpoint('logs')])
        
    value  = model_1.evaluate(test_img,test_lab,verbose=0)
    print("The network accuracy is ",value)
    prediction_model = model_1.predict(test_img)
    return model_1


pre_net(train_images,train_labels_hot,test_images,test_labels_hot,model_net,weights_dir,class_weights,10,8,opti)

