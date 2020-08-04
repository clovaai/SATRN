"""
Re-implementaition of "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition, Shi et al."
https://arxiv.org/abs/1507.05717

Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import tensorflow as tf

from networks.layers import \
        rnn_layers, conv_layer, pool_layer, dense_layer, ConvParams
from networks.Network import Network


class CRNN(Network):

    def __init__(self, FLAGS, out_charset):
        """
        """
        super().__init__(FLAGS, out_charset)

        # Set loss function
        self.loss_fn = 'ctc_loss'
        self.rnn_size = 2**8

    def preprocess_image(self, image, is_train=True):
        """
        """
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5)

        # Resize image
        image = tf.image.resize_images(
            image, [self.FLAGS.resize_hw.height, self.FLAGS.resize_hw.width],
            method=tf.image.ResizeMethod.BICUBIC)

        return image

    def get_logits(self, image, is_train, **kwargs):
        """
        """
        widths = tf.ones(tf.shape(image)[0],
                         dtype=tf.int32) * tf.shape(image)[2]
        features, sequence_length = self._convnet_layers(
            image, widths, is_train)
        features = rnn_layers(features,
                              sequence_length,
                              self.rnn_size,
                              use_projection=True)
        logits = dense_layer(features, len(self.out_charset) + 1, name='logits')

        return logits, sequence_length

    def _convnet_layers(self, inputs, widths, is_train):
        """
        Build convolutional network layers attached to the given input tensor
        """
        # Conv params : Filters K  Stride Padding  Bias   BN    Name
        conv_params = \
            [ConvParams(64, 3, (1, 1), 'same', True, False, 'conv1'),
             ConvParams(128, 3, (1, 1), 'same', True, False, 'conv2'),
             ConvParams(256, 3, (1, 1), 'same', True, False, 'conv3'),
             ConvParams(256, 3, (1, 1), 'same', True, False, 'conv4'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv5'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv6'),
             ConvParams(512, 2, (1, 1), 'valid', True, False, 'conv7')]

        with tf.variable_scope("convnet"):
            conv1 = conv_layer(inputs, conv_params[0], is_train)
            pool1 = pool_layer(conv1, 2, 'valid', 'pool1')

            conv2 = conv_layer(pool1, conv_params[1], is_train)
            pool2 = pool_layer(conv2, 2, 'valid', 'pool2')

            conv3 = conv_layer(pool2, conv_params[2], is_train)

            conv4 = conv_layer(conv3, conv_params[3], is_train)
            pool3 = pool_layer(conv4, 1, 'valid', 'pool3', wstride=1)

            conv5 = conv_layer(pool3, conv_params[4], is_train)

            conv6 = conv_layer(conv5, conv_params[5], is_train)
            pool4 = pool_layer(conv6, 1, 'valid', 'pool4', wstride=1)

            conv7 = conv_layer(pool4, conv_params[6], is_train)

            features = tf.squeeze(conv7, axis=1, name='features')

            sequence_length = widths // 4 - 1
            sequence_length = tf.reshape(sequence_length, [-1], name='seq_len')

            return features, sequence_length
