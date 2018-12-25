#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#import matplotlib.pyplot as plt
import numpy as np 
import scipy
import pandas as pd
import numpy as np
from svmutil import *
import time
##import Line


# In[4]:


(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0],28*28)
test_images = test_images.reshape(test_images.shape[0],28*28)
#train_labels = np.concatenate((train_labels,test_labels),axis=0)
#train_images
# In[6]:
#change to list for input into libsvm
train_labels = train_labels.tolist()
test_labels = test_labels.tolist()

#PCA for the train and test
num_components_PCA = 40;
pca = PCA(num_components_PCA)
pca.fit(train_images)
train_img_PCA =  pca.transform(train_images)
test_img_PCA = pca.transform(test_images)
train_PCA = train_img_PCA.tolist()
test_PCA = test_img_PCA.tolist()

# In[7]:



# In[8]:


#LDA for train and test data  
lda = LDA()
#num_components_LDA = 50;
lda = LDA(n_components = 9)
lda.fit(train_images,train_labels)
#train_img_LDA = lda.fit(train_images, train_labels).transform(train_images)
train_img_LDA = lda.transform(train_images)
test_img_LDA = lda.transform(test_images)
train_LDA = train_img_LDA.tolist()
test_LDA = test_img_LDA.tolist()
selection_val = 1 #1=polynomial 2:linear,3 rbf

# In[9]:
#SVM RBF,Linear and polynomial kerenels with PCA and LDA
print("Starting SVM")
if (selection_val == 1):
	start_time = time.time()

	svm_model_polynomial_LDA = svm_train(train_labels,train_LDA,'-t 1')
	label_pred,accu,dec_vals = svm_predict(test_labels,test_LDA,svm_model_polynomial_LDA)
	label_pred = [int(tmp) for tmp in label_pred]
	lbel_pred = np.asarray(label_pred)
	test_labels = np.asarray(test_labels)
	acc = np.mean((label_pred==test_labels)*1)
	print("Accuracy poly LDA",acc)# 91.96
	LDA_poly = acc
	end_time = time.time()
	print("The time taken is ",(end_time-start_time)/60)

	start_time =time.time()
	svm_model_polynomial_PCA = svm_train(train_labels,train_PCA,'-t 1')
	label_pred,accu,dec_vals = svm_predict(test_labels,test_PCA,svm_model_polynomial_PCA)
	label_pred = [int(tmp) for tmp in label_pred]
	lbel_pred = np.asarray(label_pred)
	test_labels = np.asarray(test_labels)
	acc = np.mean((label_pred==test_labels)*1)
	print("Accuracy poly PCA",acc)#for 40 98.11%
	end_time = time.time()
	print("The time taken is ",(end_time-start_time)/60)
elif(selection_val == 2):
	start_time =time.time()
	svm_model_linear_LDA = svm_train(train_labels,train_LDA,'-t 0')
	label_pred,accu,dec_vals = svm_predict(test_labels,test_LDA,svm_model_linear_LDA)
	label_pred = [int(tmp) for tmp in label_pred]
	lbel_pred = np.asarray(label_pred)
	test_labels = np.asarray(test_labels)
	acc = np.mean((label_pred==test_labels)*1)
	print("Accuracy Linear LDA ",acc)
	LDA_lin = acc #89.33
	end_time = time.time()
	print("The time taken is ",(end_time-start_time)/60)

	start_time =time.time()
	svm_model_linear_PCA = svm_train(train_labels,train_PCA,'-t 0')
	label_pred,accu,dec_vals = svm_predict(test_labels,test_PCA,svm_model_linear_PCA)
	label_pred = [int(tmp) for tmp in label_pred]
	lbel_pred = np.asarray(label_pred)
	test_labels = np.asarray(test_labels)
	acc = np.mean((label_pred==test_labels)*1)
	print("Accuracy Liner PCA",acc)
	PCA_lin = acc
	end_time = time.time()
	print("The time taken is ",(end_time-start_time)/60)
elif(selection_val==3):
	start_time =time.time()
	svm_model_rbf_LDA = svm_train(train_labels,train_LDA,'-t 2')
	label_pred,accu,dec_vals = svm_predict(test_labels,test_LDA,svm_model_rbf_LDA)
	label_pred = [int(tmp) for tmp in label_pred]
	lbel_pred = np.asarray(label_pred)
	test_labels = np.asarray(test_labels)
	acc = np.mean((label_pred==test_labels)*1)
	print("Accuracy for LDA RBF",acc)
	LDA_rbf = acc#92.54%
	end_time = time.time()
	print("The time taken is ",(end_time-start_time)/60)

	start_time =time.time()
	svm_model_rbf_PCA = svm_train(train_labels,train_PCA,'-t 2')
	label_pred,accu,dec_vals = svm_predict(test_labels,test_PCA,svm_model_rbf_PCA)
	label_pred = [int(tmp) for tmp in label_pred]
	lbel_pred = np.asarray(label_pred)
	test_labels = np.asarray(test_labels)
	acc = np.mean((label_pred==test_labels)*1)
	print("Accuracy for PCA rbf",acc)
	PCA_rbf = acc
	end_time = time.time()
	print("The time taken is ",(end_time-start_time)/60)
else:
	print("wrong selection value ")



