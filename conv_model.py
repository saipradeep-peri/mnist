# -*- coding: utf-8 -*-
"""
@author: SaiPradeep
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# import the train data downloaded from kaggle mnist dataaset
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv").values
test_images = np.multiply(test_data,1.0/255.0)

batch_size = 50
validation_size = 2000
n_labels = 10
labels = train_data[[0]].values.ravel()
ohem = np.zeros((labels.shape[0],n_labels))
ohem[np.arange(labels.shape[0]),labels] = 1

#converting the labels to one hot encoded matrix
labels = ohem
train_data.drop(labels = "label", axis = 1, inplace = True)
images = train_data.values
images = np.multiply(images,1.0/255.0)

train_images = images[validation_size:]
validation_images = images[:validation_size]
train_label = labels[validation_size:]
validation_label = labels[:validation_size]

print(train_images.shape)

learning_rate = 1e-4
labels = 10
graph = tf.Graph()
index_epoch = 0
num_images = train_images.shape[0]
epochs_num = 0

# creating a convolution model with 2 conv layers and 1 fully connected layer
with graph.as_default() :
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])
    x_image = tf.reshape(x,[-1,28,28,1])
    def Weights(shape):
        initial = tf.truncated_normal(shape,stddev = 0.1)
        return tf.Variable(initial)
    
    def Biases(shape):
        initial = tf.constant(0.1,shape =shape)
        return tf.Variable(initial)
    
    def conv2d(x,w):
        return tf.nn.conv2d(x,w,strides=[1,1,1,1], padding='SAME')
    
    def max_pool(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    
    def next_batch(batch_size):
        global index_epoch
        global epochs_num
        global train_images
        global train_label
        
        start = index_epoch
        index_epoch += batch_size
        if index_epoch > num_images:
            epochs_num += 1
            perm = np.arange(num_images)
            np.random.shuffle(perm)
            train_images = train_images[perm]
            train_label = train_label[perm]
            start = 0
            index_epoch = batch_size
            assert batch_size <= num_images
        end = index_epoch
        return train_images[start:end],train_label[start:end]
        

    W_layer1 = Weights([5,5,1,32])
    b_layer1 = Biases([32])
    h_layer1 = tf.nn.relu(conv2d(x_image,W_layer1) + b_layer1)
    h_pool1 = max_pool(h_layer1)
    
    W_layer2 = Weights([5,5,32,64])
    b_layer2 = Biases([64])
    h_layer2 = tf.nn.relu(conv2d(h_pool1,W_layer2) + b_layer2)
    h_pool2 = max_pool(h_layer2)
    
    W_fc1 = Weights([7*7*64,1024])
    b_fc1 = Biases([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    
    W_fc2 = Weights([1024,10])
    b_fc2 = Biases([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    predict = tf.argmax(y_conv,1)
#    tf.summary.scalar("cost", cost)
 #   tf.summary.scalar("accuracy", accuracy)
    
  #  summary_op = tf.summary.merge_all()


with tf.Session(graph = graph) as sess:
    sess.run(tf.global_variables_initializer())
    
#    writer = tf.summary.FileWriter('E:\python workspace\logs',graph=tf.get_default_graph())
    
    for i in range(20000):
        batchx ,batchy = next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batchx,y:batchy,keep_prob:0.6})
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batchx,y:batchy,keep_prob:1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        
    validation_accuracy = accuracy.eval(feed_dict={x:validation_images,y:validation_label,keep_prob:1.0})
    print("validation_accuracy %g"%(validation_accuracy))
    
    #print("test accuracy %g"%accuracy.eval(feed_dict={x:test_features, y: test_label, keep_prob: 1.0})) 
    
    predict_labels = np.zeros(test_images.shape[0])
    for i in range(0,test_images.shape[0]//batch_size):
        predict_labels[i*batch_size : (i+1)*batch_size] = predict.eval(feed_dict={x: test_images[i*batch_size : (i+1)*batch_size], keep_prob:1.0})

    np.savetxt('data/submission.csv', 
           np.c_[range(1,len(test_images)+1),predict_labels], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '',fmt='%d')
    
    
    
    