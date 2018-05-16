from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import random
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)  # Level of info shown on output

def cnn_model_fn(features, labels, mode):
    dropout_rate = 0.0
    learning_rate = 0.001
    """Model function for CNN."""
    # Input layer
    input_layer = features["x"]  # tf.reshape(features["x"], [-1, 98, 98, 3])
    print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Output shape: [-1, 49, 49, 20]

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=50,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layer.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="same")
    # Output shape: [-1, 25, 25, 50]

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 25*25*50])
    dense = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 
        # 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    # loss = tf.losses.softmax_cross_entropy(
    #     onehot_labels=onehot_labels, logits=logits)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

def open_and_

def train_input_fn(dataset_train):
	return dataset_train.make_one_shot_iterator().get_next()


# Load training and eval data
def main(unused_argv):
	batch_size = 100
	image_dir = "/home/ares/claudio/imagenes/final_data/original/"
	filenames_p = os.listdir(image_dir + 'p/')
	filenames_n = os.listdir(image_dir + 'n/')
	filenames = []
	labels = []
	for filename in filenames_p:
		filenames.append(image_dir + 'p/' + filename)
	for filename in filenames_n[0:len(filenames_p)]:
		filenames.append(image_dir + 'n/' + filename)
	random.seed(42)    
	random.shuffle(filenames)
	for filename in filenames:
		if filename.split('/')[-2] == 'p':
		    labels.append(1)
		elif filename.split('/')[-2] == 'n':
		    labels.append(0)
	filenames = tf.constant(filenames)
	labels = tf.constant(labels)
	total = filenames.shape[0].value
	filenames_train = filenames[:int(total*0.7)]
	labels_train = labels[:int(total*0.7)]
	filenames_eval = filenames[int(total*0.7):]
	labels_eval = labels[int(total*0.7):]
	dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
	dataset_train = dataset_train.map(parse_function)
	dataset_train = dataset_train.repeat().batch(batch_size)
	dataset_eval = tf.data.Dataset.from_tensor_slices((filenames_eval, labels_eval))
	dataset_eval = dataset_eval.map(parse_function)
	dataset_eval = dataset_eval.repeat().batch(batch_size)
    # Create the Estimator
	crack_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="/home/ares/claudio/crack_convnet_model")
	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
    	tensors=tensors_to_log, every_n_iter=50)
	# Train the model
	crack_classifier.train(input_fn=lambda: train_input_fn(dataset_train),
		steps=20000, hooks=[logging_hook])

if __name__ == "__main__":
    tf.app.run()
