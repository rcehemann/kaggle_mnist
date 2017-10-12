# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:10:21 2017

@author: blenderherad

tensorflow NN solution to taxi problem
"""

import tensorflow as tf
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA, KernelPCA
import os
from scipy import ndimage

tf.logging.set_verbosity(tf.logging.INFO)


                    

#%%
###############################################################################
    # set up the neural network 
###############################################################################
# convolutional neural network with architecture
# inputs: image in 28x28x1 format (1 channel)
# conv layer
# pool layer
# conv layer
# pool layer
# dense (fully connected) layer
# dense layer for outputs
def TF_CNN(features, labels, mode):
    
        # create first hidden layer; reshape flat input to image batch_size x 28px x 28px x 1 channel
        conv_1 = tf.layers.conv2d(inputs      = tf.reshape(features['x'], [-1,28,28,1]),
                                  filters     = 64,
                                  kernel_size = [5,5],
                                  padding     = "same",
                                  activation  = tf.nn.relu)
        
        # pool outputs of first convolution. pool_size prevents overlap
        pool_1 = tf.layers.max_pooling2d(inputs    = conv_1,
                                         pool_size = [2,2],
                                         strides   = 2)
        
        #repeat
        conv_2 = tf.layers.conv2d(inputs       = pool_1,
                                  filters      = 128,
                                  kernel_size  = [5,5],
                                  padding      = "same",
                                  activation   = tf.nn.relu)
        
        pool_2 = tf.layers.max_pooling2d(inputs    = conv_2,
                                         pool_size = [2,2],
                                         strides   = 2)
        
        # fully-connected layers for learning on features extracted by convolution
        dense_1  = tf.layers.dense(inputs         = tf.reshape(pool_2, [-1, 7*7*128]),
                                   units          = 1024,
                                   activation     = tf.nn.relu)
        
        #dropout regularization
        dropout_1 = tf.layers.dropout(inputs   = dense_1,
                                       rate     = 0.5,
                                       training = mode == tf.estimator.ModeKeys.TRAIN)
        
        # logits for softmax / argmax
        logits     = tf.layers.dense(inputs = dropout_1, units=10)
        
        predictions = {'classes': tf.argmax(logits, axis=1),
                       'probabilities': tf.nn.softmax(logits, name='softmax_probs')}
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
        one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
            train_op  = optimizer.minimize(loss = loss,
                                           global_step = tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
        
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels = labels, predictions = predictions['classes'])}
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)


# normalize by row
def norm_by_row(row):
    row  = row.values.reshape((28,28)).astype(float)
    for j in range(28):
        if np.any(row[:,j]):
            #row[:,j] *= 1./np.max(row[:,j])
            row[:,j] *= 1./255.
            
    row  = row.reshape((784,))
    return row

# shift images to center of mass
def shift_to_com(row):
    row = row.values.reshape(28,28)
    shx = np.array([14,14])-np.array(list(map(int,ndimage.center_of_mass(row))))
    row = np.roll(row, shx[0], axis=0)
    row = np.roll(row, shx[1], axis=1)
    return row.reshape((784,))    

def main(unused_argv):
    
    if os.path.isfile('train_t.csv'):
        df = pd.read_csv('train_t.csv')
        tdf= pd.read_csv('test_t.csv')
        Yf = pd.read_csv('labels.csv')
    else:
        df = pd.read_csv('train.csv')
        tdf = pd.read_csv('test.csv')
    
        # rotate images to align second moment in the vertical direction
        ## try normalizing each row and column before rotating. possible errors from 
        ## extra weight coming from heavy pixels at one end of the number or the other
        #Y = np.array([[1. if i == value else 0. for i in range(10)] for value in df['label'].values])
        Yf = pd.DataFrame(df.label.values, columns=['label'])
        df.drop('label', axis=1, inplace=True)
    
        # apply fns to training data
        #df = df.apply(lambda row: norm_by_row(row), axis=1)
        #df = df.apply(lambda row: shift_to_com(row), axis=1)
        
        # apply fns to testing data
        #tdf = tdf.apply(lambda row: norm_by_row(row), axis=1)
        #tdf = tdf.apply(lambda row: shift_to_com(row), axis=1)
        
        #Yf = pd.DataFrame(Y, columns=['lab0', 'lab1', 'lab2', 'lab3', 'lab4', 'lab5', 'lab6', 'lab7', 'lab8', 'lab9'])
        Yf.to_csv('labels.csv', index=False)
        df.to_csv('train_t.csv', index=False)
        tdf.to_csv('test_t.csv', index=False)

    Y = Yf.label.values.astype(np.int32)
    X = df.values.astype(np.float32)
    X_test  = tdf.values.astype(np.float32)
#    scaler = StandardScaler()
#    scaler.fit(X)
#    X = scaler.transform(X)
#    X_test  = scaler.transform(X_test)


    epochs        = 100
    batch_size    = 500
        
    mnist_classifier = tf.estimator.Estimator(model_fn = TF_CNN, model_dir = "./cnn_checkpoints")
    
    #tensors_to_log = {'probabilities': 'softmax_probs'}
    #logging_hook   = {tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 50)}
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": X},  y = Y, batch_size = batch_size, num_epochs = epochs, shuffle=True)
    mnist_classifier.train(input_fn = train_input_fn, steps = 10000)#, hooks = [logging_hook])
    
    eval_input_fn   = tf.estimator.inputs.numpy_input_fn(x = {"x": X}, y = Y, num_epochs = 1, shuffle=False)
    eval_results    = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)
    
    pred_input_fn   = tf.estimator.inputs.numpy_input_fn(x = {"x": X_test}, num_epochs = 1, shuffle=False)
    pred_results    = list(mnist_classifier.predict(input_fn = pred_input_fn))
    
    predictions     = [p['classes'] for p in pred_results]
    output = pd.DataFrame({'ImageId':(tdf.index.values+1), 'Label':predictions})
    print(output.head())
    output.to_csv('tensorflow_cnn.csv', index=False)     
                                                      
    #############################################
    ##         Training and prediction
    #############################################
    # start the TF session
        
#    print("beginning tensorflow session...")
#    with tf.Session() as sess:
#        #initilize
#        sess.run(init_op)
#        total_batch = num_passes*int(len(df) / batch_size)
#        epoch_cost = [0] * epochs
#        for epoch in range(epochs):
#            avg_cost = 0; avg_pen=0
#            for i in range(total_batch):
#                
#                #sample  = list(np.random.randint(0,len(X)-1,size=[batch_size]))
#                #X_train = X[sample]
#                #Y_train = Y[sample]
#                X_train = X[i*batch_size:(i+1)*batch_size]
#                Y_train = Y[i*batch_size:(i+1)*batch_size]
#    
#                _, c, pen = sess.run([opt, cost, norm_penalty], feed_dict={x: X_train, y: Y_train})
#                avg_cost += c
#                avg_pen  += pen
#            print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost/total_batch), "L2-loss=", "{:.5f}".format(avg_pen/total_batch))
#            epoch_cost[epoch] = avg_cost
#            #for i in range(3):
#            #print ("       true value:", np.argmax(Y_samp[i]), "predicted value", np.argmax(p[i]))
        
#        print(sess.run(accuracy, feed_dict={x: X, y: Y}))
#        print("predicting on test set...")
#        predictions = sess.run(tf.argmax(y_,axis=1), feed_dict={x: X_test})
         #%%
    #############################################
    ##               OUTPUT
    #############################################
    
    
#    output = pd.DataFrame({'ImageId':(tdf.index.values+1), 'Label':predictions})
#    print(output.head())
#    output.to_csv('tensorflow_cnn.csv', index=False) 
if __name__ == "__main__":
    tf.app.run()