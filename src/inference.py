"""
Copyright (c) 2020-present NAVER Corp.
MIT license

Usage:
python inference.py --config_file=<config_file_path> --image_file=<image_file_path>
"""

import cv2
import fire
import numpy as np
import tensorflow as tf
from flags import Flags
from utils import get_network, load_charset, get_string, get_init_trained, adjust_string


def inference(config_file, image_file):
    """ Run text recognition network on an image file.
    """
    # Get config
    FLAGS = Flags(config_file).get()
    out_charset = load_charset(FLAGS.charset)
    num_classes = len(out_charset)
    net = get_network(FLAGS, out_charset)

    if FLAGS.use_rgb:
        num_channel = 3
        mode = cv2.IMREAD_COLOR
    else:
        num_channel = 1
        mode = cv2.IMREAD_GRAYSCALE

    # Input node
    image = tf.placeholder(tf.uint8,
                           shape=[None, None, num_channel],
                           name='input_node')

    # Network
    proc_image = net.preprocess_image(image, is_train=False)
    proc_image = tf.expand_dims(proc_image, axis=0)
    proc_image.set_shape(
        [None, FLAGS.resize_hw.height, FLAGS.resize_hw.width, num_channel])
    logits, sequence_length = net.get_logits(proc_image,
                                             is_train=False,
                                             label=None)
    prediction, log_prob = net.get_prediction(logits, sequence_length)
    prediction = tf.sparse_to_dense(sparse_indices=prediction.indices,
                                    sparse_values=prediction.values,
                                    output_shape=prediction.dense_shape,
                                    default_value=num_classes,
                                    name='output_node')

    # Restore
    restore_model = get_init_trained()
    sess = tf.Session()
    restore_model(sess, FLAGS.eval.model_path)

    # Run
    img = cv2.imread(image_file, mode)
    img = np.reshape(img, [img.shape[0], img.shape[1], num_channel])
    predicted = sess.run(prediction, feed_dict={image: img})
    string = get_string(predicted[0], out_charset)
    string = adjust_string(string, FLAGS.eval.lowercase,
                           FLAGS.eval.alphanumeric)
    print(string)

    return string


if __name__ == '__main__':
    fire.Fire(inference)
