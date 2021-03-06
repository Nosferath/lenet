""" Convolutional Neural Network.
Modified version of Aymeric's Convolutional Neural Network 
(https://github.com/aymericdamien/TensorFlow-Examples/)
"""

from __future__ import division, print_function, absolute_import
from load_data import load_data
import tensorflow as tf

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 32  # 128
display_step = 10

# Import crack_data
image_dir = "/home/inti/Desktop/Claudio/final_data/original/"
images_train, images_eval, labels_train, labels_eval = load_data(
            batch_size, image_dir, one_hot=True)
dict_images_train = {'x': images_train}
dataset_train = tf.data.Dataset.from_tensor_slices(
        (images_train, labels_train))
dataset_train = dataset_train.batch(batch_size)
iterator_train = dataset_train.make_initializable_iterator()

# Network Parameters
num_input = 96*96*3 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 1 # Dropout, probability to keep units

sess = tf.Session()

_data = tf.placeholder(tf.float32, [None, 96, 96, 3])
_labels = tf.placeholder(tf.float32, [None, num_classes])

sess.run(iterator_train.initializer, feed_dict={_data: images_train,
                                                _labels: labels_train})

# tf Graph input
X, Y = iterator_train.get_next()
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 96, 96, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 channels, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([24*24*64, 500])),
    # 500 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([500, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([500])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
batch_x, batch_y = iterator_train.get_next()
# Start training

# Run the initializer
sess.run(init)

for step in range(1, num_steps+1):  #mnist.train.next_batch(batch_size)
    #batch_x, batch_y = batch_x.eval(session=sess), batch_y.eval(session=sess)
    # Run optimization op (backprop)
    try:
        sess.run(train_op, feed_dict={keep_prob: 0.8})
    except tf.errors.OutOfRangeError:
        sess.run(iterator_train.initializer,
                 feed_dict={_data: images_train,
                            _labels: labels_train})
        sess.run(train_op, feed_dict={keep_prob: 0.8})
    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={keep_prob: 1.0})
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

print("Optimization Finished!")

# Calculate accuracy for 256 MNIST test images
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: images_eval[:256],  #mnist.test.images[:256],
                                  Y: labels_eval[:256],  #mnist.test.labels[:256],
                                  keep_prob: 1.0}))