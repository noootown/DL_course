import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read mnist data from tensorflow example
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# config
lr =  # learning rate
train_steps =
batch_size =
logs_path = 'tensorboard/'
n_features = x_train.shape[1] # 784
n_labels = y_train.shape[1] # 10

with tf.Session(config=tf.ConfigProto()) as sess:
    with tf.name_scope('inputs'):
    
    with tf.name_scope('labels'):

    # build variables
    with tf.name_scope('params'):
    
    # build model
    with tf.name_scope('model'):
    
    # define loss
    with tf.name_scope('loss'):
    
    # Gradient Descent
    with tf.name_scope('gd'):
    
    with tf.name_scope('accuracy'):
    
    # start run
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    
    # display data
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, graph = sess.graph)
    
    for step in xrange(train_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_batch})
    
        if step % 50 == 0:
            l, summary = sess.run([loss, merged], feed_dict = {x: x_batch, y: y_batch})
            ac = sess.run(acc, feed_dict={x: x_test, y: y_test})
            writer.add_summary(summary, step)
            
            print("Accuracy: %.2f, Loss: %.2f" % (ac, l))
