'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 10000
display_step = 50

DATA_PATH = '../machine-learning-ex1/ex1/ex1data1.csv'

my_data = numpy.genfromtxt(DATA_PATH, delimiter=',').astype(numpy.float32)

train_X= numpy.array([ i[0] for i in my_data])
train_Y= numpy.array([ i[1] for i in my_data])

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum( tf.pow(pred-Y, 2) ) / ( 2 * n_samples )
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:

    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):

        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict = { X: x, Y: y })

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W = ", sess.run(W), "b=", sess.run(b), '\n'

