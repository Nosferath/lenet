from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import random
import cv2
import numpy as np
import sklearn.model_selection as sk
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)  # Level of info shown on output

def cnn_model_fn(features, labels, mode):
    dropout_rate = 0.0
    learning_rate = 0.001
    """Model function for CNN."""
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 96, 96, 3])
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Output shape: [-1, 48, 48, 20]

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=50,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="same")
    # Output shape: [-1, 24, 24, 50]

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 24*24*50])
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

def input_fn(images, labels, batch_size):
    dict_images = {'x': images}
    dataset = tf.data.Dataset.from_tensor_slices((dict(dict_images), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


# Load training and eval data
def main(unused_argv):
    batch_size = 5
    image_dir = "/home/srmilab/Claudio/RefineNet/refinenet-image-segmentation/images/110_0342/final_data/original/"
    #image_dir = "/home/ares/claudio/imagenes/final_data/original/"
    #image_dir = "/home/claudio/segmentacion/imagenes/110_0342/final_data/original/"
    filenames_p = os.listdir(image_dir + 'p/')
    filenames_n = os.listdir(image_dir + 'n/')
    images = []
    labels = []
    for filename in filenames_p:
        images.append(cv2.imread(image_dir + 'p/' + filename))
        labels.append(1)
    for filename in filenames_n[0:len(filenames_p)]:
        images.append(cv2.imread(image_dir + 'n/' + filename))
        labels.append(0)
    # Converting list([98,98,3]) to array(n, 98, 98, 3)
    images = np.float32(np.stack(images))
    labels = np.int32(np.array(labels))
    images_train, images_eval, labels_train, labels_eval = sk.train_test_split(
        images, labels, test_size=0.3, random_state=42)
    # Create the Estimator
    crack_classifier = tf.estimator.Estimator(
        #model_fn=cnn_model_fn, model_dir="/home/claudio/segmentacion/crack_convnet_model")
        #model_fn=cnn_model_fn, model_dir="/home/ares/claudio/crack_convnet_model")
        model_fn=cnn_model_fn, model_dir="/home/srmilab/Claudio/crack_convnet_model")
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # Train the model 
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": images_train},
        y=labels_train,
        batch_size=batch_size,
        num_epochs=50,
        shuffle=True)
    crack_classifier.train(
        input_fn=lambda: input_fn(images_train, labels_train, batch_size),
        steps=20000,
        hooks=[logging_hook])
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": images_eval},
        y=labels_eval,
        num_epochs=1,
        shuffle=False)
    eval_results = crack_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
if __name__ == "__main__":
    tf.app.run()
