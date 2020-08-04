"""
Re-implementaition of "AON: Towards Arbitrarily-Oriented Text Recognition, Cheng at el."
https://arxiv.org/abs/1711.04226

Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import math
import tensorflow as tf

from networks.layers import \
        rnn_layer, attention_decoder, conv_layer, pool_layer, ConvParams
from networks.Network import Network


class AON(Network):

    def __init__(self, out_charset, loss_fn=None):
        """
        """
        super().__init__(out_charset)

        # Set Default Settings
        self.loss_fn = 'cross_ent'
        self.rnn_size = 2**8

    def preprocess_image(self, image, is_train=True):
        """
        """
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5)

        # Rotation augmenatation
        if is_train:
            image = tf.image.resize_images(image, (140, 140),
                                           method=tf.image.ResizeMethod.BICUBIC)

            up_pad = tf.concat([image[0:1, :, :] for _ in range(35)], axis=0)
            up_pad += tf.random_normal(tf.shape(up_pad), stddev=1e-2)
            down_pad = tf.concat([image[-1:, :, :] for _ in range(35)], axis=0)
            down_pad += tf.random_normal(tf.shape(down_pad), stddev=1e-2)
            image = tf.image.resize_images(image, (70, 70),
                                           method=tf.image.ResizeMethod.BICUBIC)
            left_pad = tf.concat([image[:, 0:1, :] for _ in range(35)], axis=1)
            left_pad += tf.random_normal(tf.shape(left_pad), stddev=1e-2)
            right_pad = tf.concat([image[:, -1:, :] for _ in range(35)], axis=1)
            right_pad += tf.random_normal(tf.shape(right_pad), stddev=1e-2)

            image = tf.concat([left_pad, image], axis=1)
            image = tf.concat([image, right_pad], axis=1)
            image = tf.concat([up_pad, image], axis=0)
            image = tf.concat([image, down_pad], axis=0)

            image = tf.contrib.image.rotate(image,
                                            tf.random_normal([],
                                                             mean=0.0,
                                                             stddev=0.3),
                                            interpolation='BILINEAR')
            image = tf.image.central_crop(image, 0.65)

        # Resize image
        image = tf.image.resize_images(image, (100, 100),
                                       method=tf.image.ResizeMethod.BICUBIC)

        return image

    def get_logits(self, image, is_train, **kwargs):
        """
        """
        image = tf.reshape(image, [-1, 100, 100, 1])

        # BCNN
        features = self._bcnn(image, is_train)
        assert features.get_shape()[1:] == (26, 26, 256)

        # AON
        features, clue = self._aon(features, is_train)
        assert features.get_shape()[1:] == (4, 23, 512)
        assert clue.get_shape()[1:] == (4, 23, 1)

        # FG
        features = tf.reduce_sum(features * clue, axis=1)
        features = tf.nn.tanh(features)
        assert features.get_shape()[1:] == (23, 512)

        # LSTM
        features = tf.transpose(features, [1, 0, 2], name='time_major')
        features = rnn_layer(features, None, self.rnn_size, 'lstm')
        logits, weights = attention_decoder(features, kwargs['label'],
                                            len(self.out_charset),
                                            self.rnn_size, is_train,
                                            self.FLAGS.label_maxlen)

        sequence_length = None

        return logits, sequence_length

    def _bcnn(self, inputs, is_train):
        """
        """
        bcnn_params = \
            [ConvParams(64, 3, (1, 1), 'same', False, True, 'conv1'),
             ConvParams(128, 3, (1, 1), 'same', False, True, 'conv2'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'conv3'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'conv4')]

        assert inputs.get_shape()[1:] == (100, 100, 1)

        with tf.variable_scope("bcnn"):
            conv1 = conv_layer(inputs, bcnn_params[0], is_train)
            pool1 = pool_layer(conv1, 2, 'valid', 'pool1')

            conv2 = conv_layer(pool1, bcnn_params[1], is_train)
            conv2 = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]])
            pool2 = pool_layer(conv2, 2, 'valid', 'pool2')

            conv3 = conv_layer(pool2, bcnn_params[2], is_train)

            features = conv_layer(conv3, bcnn_params[3], is_train)

        return features

    def _shared_cnn(self, inputs, is_train, reuse=False):
        """
        """
        shared_cnn_params = \
            [ConvParams(512, 3, (1, 1), 'same', False, True, 'conv1'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv2'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv3'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv4'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv5')]

        with tf.variable_scope("shared_cnn", reuse=reuse):
            conv1 = conv_layer(inputs, shared_cnn_params[0], is_train)
            conv1 = tf.pad(conv1, [[0, 0], [1, 1], [0, 0], [0, 0]])
            pool1 = pool_layer(conv1, 2, 'valid', 'pool1', wstride=1)

            conv2 = conv_layer(pool1, shared_cnn_params[1], is_train)
            conv2 = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]])
            pool2 = pool_layer(conv2, 2, 'valid', 'pool2', wstride=1)

            conv3 = conv_layer(pool2, shared_cnn_params[2], is_train)
            pool3 = pool_layer(conv3, 2, 'valid', 'pool3', wstride=1)

            conv4 = conv_layer(pool3, shared_cnn_params[3], is_train)
            pool4 = pool_layer(conv4, 2, 'valid', 'pool4', wstride=1)

            conv5 = conv_layer(pool4, shared_cnn_params[4], is_train)
            pool5 = pool_layer(conv5, 2, 'valid', 'pool5', wstride=1)

            features = tf.reshape(pool5, (-1, 23, 512))

        return features

    def _clue_network(self, inputs, is_train):
        """
        """
        clue_network_params = \
            [ConvParams(512, 3, (1, 1), 'same', False, True, 'conv1'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv2')]

        weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
        bias_initializer = tf.constant_initializer(value=0.0)

        assert inputs.get_shape()[1:] == (26, 26, 256)

        with tf.variable_scope("clue_network"):
            conv1 = conv_layer(inputs, clue_network_params[0], is_train)
            conv1 = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]])
            pool1 = pool_layer(conv1, 2, 'valid', 'pool1')

            conv2 = conv_layer(pool1, clue_network_params[1], is_train)
            conv2 = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]])
            pool2 = pool_layer(conv2, 2, 'valid', 'pool2')

            features = tf.reshape(pool2, (-1, 64, 512))
            features = tf.transpose(features, perm=[0, 2, 1])
            features = tf.layers.dense(features,
                                       23,
                                       kernel_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       activation=tf.nn.relu,
                                       name='length_dense')
            features = tf.contrib.layers.dropout(features,
                                                 keep_prob=0.8,
                                                 is_training=is_train)

            features = tf.transpose(features, perm=[0, 2, 1])
            features = tf.layers.dense(features,
                                       4,
                                       kernel_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       activation=tf.nn.softmax,
                                       name='channel_dense')

            features = tf.transpose(features, perm=[0, 2, 1])
            features = tf.expand_dims(features, axis=-1)

        return features

    def _aon(self, inputs, is_train):
        """
        """
        assert inputs.get_shape()[1:] == (26, 26, 256)

        with tf.variable_scope("aon"):
            hfeatures = self._shared_cnn(inputs, is_train, reuse=False)
            vfeatures = self._shared_cnn(tf.contrib.image.rotate(
                inputs, math.pi / 2),
                                         is_train,
                                         reuse=True)

            hfeatures = tf.transpose(hfeatures,
                                     perm=[1, 0, 2],
                                     name='h_time_major')
            hfeatures = rnn_layer(hfeatures, None, self.rnn_size, 'hbdrnn')
            vfeatures = tf.transpose(vfeatures,
                                     perm=[1, 0, 2],
                                     name='v_time_major')
            vfeatures = rnn_layer(vfeatures, None, self.rnn_size, 'vbdrnn')

            features = (hfeatures, tf.reverse(hfeatures, axis=[0])) + \
                       (vfeatures, tf.reverse(vfeatures, axis=[0]))
            features = tf.stack(features, axis=1)
            features = tf.transpose(features, [2, 1, 0, 3])

            clue = self._clue_network(inputs, is_train)

        return features, clue
