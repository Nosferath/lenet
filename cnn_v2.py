""" TensorFlow Dataset API.
Modified version of Aymeric's Dataset API example. 
(https://github.com/aymericdamien/TensorFlow-Examples/)
"""
from __future__ import print_function

import tensorflow as tf
import numpy as np
from load_data import load_data
# Import MNIST data (Numpy format)
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
num_steps = 20000
batch_size = 32
display_step = 10

# Import crack_data
image_dir = "/home/inti/Desktop/Claudio/final_data/original/"
images_train, images_eval, labels_train, labels_eval = load_data(
            batch_size, image_dir, one_hot=True)
images_train = np.reshape(images_train, [-1, 96*96*3])

# Network Parameters
n_input = 96*96*3 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units

sess = tf.Session()

# Create a dataset tensor from the images and the labels
dataset = tf.data.Dataset.from_tensor_slices(
    (images_train, labels_train))
# Create batches of data
dataset = dataset.repeat().batch(batch_size)
# Create an iterator, to go over the dataset
iterator = dataset.make_initializable_iterator()
# It is better to use 2 placeholders, to avoid to load all data into memory,
# and avoid the 2Gb restriction length of a tensor.
_data = tf.placeholder(tf.float32, [None, n_input])
_labels = tf.placeholder(tf.float32, [None, n_classes])
# Initialize the iterator
sess.run(iterator.initializer, feed_dict={_data: images_train,
                                          _labels: labels_train})

# Neural Net Input
X, Y = iterator.get_next()


# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Note that a few elements have changed (usage of sess run).

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 96, 96, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 500)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out

def cnn_model_fn(x, n_classes, dropout, reuse, is_training):
    """Model function for CNN."""
    # Input layer
    x = tf.reshape(x, [-1, 96, 96, 3])
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(x, 20, 5, activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    # Output shape: [-1, 48, 48, 20]

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(pool1, 50, 5, activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    # Output shape: [-1, 24, 24, 50]

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(pool2, 100, 5, activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
    # Output shape: [-1, 12, 12, 100]

    # Dense Layer #1
    pool3_flat = tf.contrib.layers.flatten(pool3)
    dense1 = tf.layers.dense(pool3_flat, 2000)
    dropout1 = tf.layers.dropout(dense1, rate=dropout, training=is_training)

    # Dense Layer #2
    dense2 = tf.layers.dense(dropout1, 1000)
    dropout2 = tf.layers.dropout(dense2, rate=dropout, training=is_training)

    # Dense Layer #3
    dense3 = tf.layers.dense(dropout2, 500)
    dropout3 = tf.layers.dropout(dense3, rate=dropout, training=is_training)

    # Dense Layer #4
    dense4 = tf.layers.dense(dropout3, 100)
    dropout4 = tf.layers.dropout(dense4, rate=dropout, training=is_training)
    # Logits Layer
    logits = tf.layers.dense(dropout4, n_classes)
    out = tf.nn.softmax(logits) if not is_training else logits
    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = cnn_model_fn(X, n_classes, dropout, reuse=tf.AUTO_REUSE, is_training=True)
# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
logits_test = cnn_model_fn(X, n_classes, dropout, reuse=tf.AUTO_REUSE, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
sess.run(init)

# Training cycle
for step in range(1, num_steps + 1):

    try:
        # Run optimization
        sess.run(train_op)
    except tf.errors.OutOfRangeError:
        # Reload the iterator when it reaches the end of the dataset
        sess.run(iterator.initializer,
                 feed_dict={_data: images_train,
                            _labels: labels_train})
        sess.run(train_op)
        
    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        # (note that this consume a new batch of data)
        loss, acc = sess.run([loss_op, accuracy])
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

print("Optimization Finished!")