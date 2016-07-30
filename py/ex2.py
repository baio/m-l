'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy as np

# Import MINST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

DATA_PATH = '../data/iris.csv'
my_data = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=True).astype(np.float32)
train = np.delete(my_data, 4, 1)
# print(train)
n_samples = train.shape[0]

#train_Y = numpy.array([ i[5] for i in my_data])

# Parameters
learning_rate = 0.01
training_epochs = 400
batch_size = 20
display_step = 1

# tf Graph Input
# 4 - features, 4 - 3 classes
x = tf.placeholder(tf.float32, [None, 4]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 3]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples/batch_size)
        # Loop over all batches

        for i in range(total_batch):
            batch = np.random.permutation(train)[0:batch_size,:]
            batch_xs = batch[:, 0:4]
            batch_ys =  batch[:, 4]
            onehot = np.eye(3)[batch_ys.astype(np.int)]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: onehot})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})