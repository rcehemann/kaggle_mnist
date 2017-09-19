# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:10:21 2017

@author: blenderherad

tensorflow NN solution to mnist handwriting data
"""

import tensorflow as tf
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from dateutil import parser
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
import os, math
from scipy import ndimage

n_PCA  = 10

#%%


df = pd.read_csv('train.csv')
tdf = pd.read_csv('test.csv')

if os.path.isfile('taxi.pckl'):
    
    df = pd.read_pickle('taxi.pckl')

# normalize by row
def norm_by_row(row):
    row  = np.reshape(row,(28,28))
    sums = row.sum(axis=1)
    sums[sums==0] = 1 # avoid dividing by zero    
    row  = (row.T/sums).T
    row  = np.reshape(row, (784,))
    return row

# shift images to center of mass
def shift_to_com(row):
    row = np.reshape(row,(28,28))
    shx = np.array([14,14])-np.array(list(map(int,ndimage.center_of_mass(row))))
    row = np.roll(row, shx[0], axis=0)
    row = np.roll(row, shx[1], axis=1)
    return np.reshape(row, (784,))

# rotate images to align second moment in the vertical direction
def rot_2_moment(row):
    row = np.reshape(row,(28,28))
    wgt = [(k-14)**2 for k in range(28)]
    xmo = np.sum(np.dot(row,wgt))
    ymo = np.sum(np.dot(wgt,row))
    tet = -np.arccos(ymo/np.sqrt(xmo**2+ymo**2))
    ret=row
    print(tet*180./np.pi)
    for i in range(28):
        for j in range(28):
            ret[i,j] = row[int(i*cos(tet))-int(j*sin(tet))%28, int(i*sin(tet)) + int(j*cos(tet))%28]
            
    return np.reshape(ret, (784,))

Y = np.array([[1. if i == value else 0. for i in range(10)] for value in df['label'].values])
df.drop('label', axis=1, inplace=True)

## try normalizing each row and column before rotating. possible errors from 
## extra weight coming from heavy pixels at one end of the number or the other

df = df.apply(lambda row: norm_by_row(row), axis=1)
df = df.apply(lambda row: norm_by_row(row.T), axis=1)
#df = df.apply(lambda row: rot_2_moment(row), axis=1)
df = df.apply(lambda row: shift_to_com(row), axis=1)

X = df.values.astype(float)

tdf = tdf.apply(lambda row: norm_by_row(row), axis=1)
tdf = tdf.apply(lambda row: shift_to_com(row), axis=1)
X_test  = tdf.astype(float)
#scaler = preprocessing.StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)
#X_test  = scaler.transform(X_test)

# define function to roll axes based on center of mass



    

#%%
###############################################################################
    # set up the neural network 
###############################################################################
learning_rate = 0.005
epochs        = 300
batch_size    = 100

n_input= 784 # = 28*28 pixels
hlsize = 300
n_out = 10
n_hidden_layers = 1
n_h  = [hlsize] * n_hidden_layers

h_lay = [0.] * n_hidden_layers
istddev = 1./math.sqrt(n_input)

# declare a placeholder for training data
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_out  ])

# multi-layer perceptron
def MLP(x, w, b):
    
    # create first hidden layer
    h_lay[0] = tf.nn.relu(tf.add(tf.matmul(x, w[0]), b[0]))
        
    # create N-layer MLP with RELU activation
    for i in range(1, n_hidden_layers):
        h_lay[i] = tf.nn.relu(tf.add(tf.matmul(h_lay[i-1], w[i]), b[i]))
           
    # return output layer
    return tf.nn.softmax(tf.add(tf.matmul(h_lay[-1], w[-1]), b[-1]), dim=1)   
    
    
# build lists for weights and biases
w = [tf.Variable(tf.random_normal([n_input, n_h[0]], 0, istddev))]
for i in range(1, n_hidden_layers):
    w.append(tf.Variable(tf.random_normal([n_h[i-1], n_h[i]], 0, istddev)))
w.append(tf.Variable(tf.random_normal([n_h[-1], n_out], 0, istddev)))

b = [tf.Variable(tf.random_normal([n_h[0]], 0, istddev))]
for i in range(1, n_hidden_layers):
    b.append(tf.Variable(tf.random_normal([n_h[i]], 0, istddev)))
b.append(tf.Variable(tf.random_normal([n_out], 0, istddev)))

y_ =  MLP(x, w, b)

#cost            = tf.reduce_mean(tf.reduce_sum(tf.square(tf.log(y_+1) - tf.log(y+1)),axis=1))
y_clipped = tf.clip_by_value(y_,1e-7,0.999999)

# cost is the cross-entropy

cost            = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1-y_clipped), axis=1))
# now set up an optimizer
#opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# initialization operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
accuracy           = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(y,1)), tf.float32))
    
        #sns.regplot(df['pickup_longitude'].values, df['pickup_latitude'].values, color=df['cluster'])
    #%%
    #############################################
    ##               Training and prediction
    #############################################
    # start the TF session
    
print("beginning tensorflow session...")
with tf.Session() as sess:
    #initilize
    sess.run(init_op)
    total_batch = int(len(df) / batch_size)
    epoch_cost = [0] * epochs
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            X_train = X[i*batch_size:(i+1)*batch_size]
            Y_train = Y[i*batch_size:(i+1)*batch_size]

            _, c, p = sess.run([opt, cost, y_], feed_dict={x: X_train, y: Y_train})
            avg_cost += c/total_batch
        print("Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost))
        epoch_cost[epoch] = avg_cost
        #for i in range(3):
        #print ("       true value:", np.argmax(Y_samp[i]), "predicted value", np.argmax(p[i]))
    
    print(sess.run(accuracy, feed_dict={x: X, y: Y}))
    print("predicting on test set...")
    predictions = sess.run(tf.argmax(y_,axis=1), feed_dict={x: X_test})
    
 #%%
#############################################
##               OUTPUT
#############################################


output = pd.DataFrame({'ImageId':(tdf.index.values+1), 'Label':predictions})
print(output.head())
output.to_csv('tensorflow_nn.csv', index=False)   
            
