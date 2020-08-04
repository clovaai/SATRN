"""
Re-implementaition of "Gated Recurrent Convolution Neural Network for OCR, Wang et al."
https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf

Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import tensorflow as tf

from networks.layers import \
        rnn_layers, conv_layer, pool_layer, norm_layer, dense_layer, ConvParams
from networks.Network import Network


class GRCNN(Network):

    def __init__(self, FLAGS, out_charset):
        """
        """
        super().__init__(FLAGS, out_charset)

        # Set loss function
        self.loss_fn = 'ctc_loss'

        self.rnn_size = self.FLAGS.rnn_size or 2**8

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

        conv_params = \
            [ConvParams(64, 3, (1, 1), 'same', True, False, 'conv1'),
             ConvParams(512, 2, (1, 1), 'valid', False, True, 'conv2')]
        recur_params = [{'channel': 64}, {'channel': 128}, {'channel': 256}]

        with tf.variable_scope("convnet"):
            conv1 = conv_layer(inputs, conv_params[0], is_train)
            pool1 = pool_layer(conv1, 2, 'valid', 'pool1')
            grcl1 = self._gated_recurrent_conv_layer(pool1,
                                                     recur_params[0],
                                                     is_train,
                                                     iteration=3,
                                                     name='grcl1')

            pool2 = pool_layer(grcl1, 2, 'valid', 'pool2')
            grcl2 = self._gated_recurrent_conv_layer(pool2,
                                                     recur_params[1],
                                                     is_train,
                                                     iteration=3,
                                                     name='grcl2')

            grcl2 = tf.pad(grcl2, [[0, 0], [0, 0], [1, 1], [0, 0]])
            pool3 = pool_layer(grcl2, 2, 'valid', 'pool3', wstride=1)
            grcl3 = self._gated_recurrent_conv_layer(pool3,
                                                     recur_params[2],
                                                     is_train,
                                                     iteration=3,
                                                     name='grcl3')
            grcl3 = tf.pad(grcl3, [[0, 0], [0, 0], [1, 1], [0, 0]])

            pool4 = pool_layer(grcl3, 2, 'valid', 'pool4', wstride=1)
            conv2 = conv_layer(pool4, conv_params[1], is_train)
            features = tf.squeeze(conv2, axis=1, name='features')

            sequence_length = widths // 4 + 1
            sequence_length = tf.reshape(sequence_length, [-1], name='seq_len')

            return features, sequence_length

    def _gated_recurrent_conv_layer(self, bottom, params, is_train, iteration,
                                    name):
        """
        GRCL
        """
        with tf.variable_scope(name):
            in_channel = bottom.get_shape()[-1]
            kernel_initializer = \
                tf.contrib.layers.variance_scaling_initializer()
            bias_initializer = tf.constant_initializer(value=0.0)

            # Get variables
            wgf = tf.get_variable("WGF",
                                  shape=(1, 1, in_channel, params['channel']),
                                  initializer=kernel_initializer)
            bgf = tf.get_variable("BGF",
                                  shape=(params['channel'],),
                                  initializer=bias_initializer)

            wgr = tf.get_variable("WGR",
                                  shape=(1, 1, params['channel'],
                                         params['channel']),
                                  initializer=kernel_initializer)
            bgr = tf.get_variable("BGR",
                                  shape=(params['channel'],),
                                  initializer=bias_initializer)

            wf = tf.get_variable("WF",
                                 shape=(3, 3, in_channel, params['channel']),
                                 initializer=kernel_initializer)
            bf = tf.get_variable("BF",
                                 shape=(params['channel']),
                                 initializer=bias_initializer)

            wr = tf.get_variable("WR",
                                 shape=(3, 3, params['channel'],
                                        params['channel']),
                                 initializer=kernel_initializer)
            br = tf.get_variable("BR",
                                 shape=(params['channel'],),
                                 initializer=bias_initializer)

            # Common terms
            ugf = tf.nn.conv2d(bottom,
                               wgf,
                               strides=(1, 1, 1, 1),
                               padding='VALID')
            ugf = tf.nn.bias_add(ugf, bgf)
            uf = tf.nn.conv2d(bottom, wf, strides=(1, 1, 1, 1), padding='SAME')
            uf = tf.nn.bias_add(uf, bf)

            # Ground Case
            x_t = tf.nn.relu(norm_layer(uf, is_train, 'UF/batch_norm'))

            for t in range(1, iteration + 1):
                # G(t) = sig(bn(wgf*u) + bn(wgr*x))
                G_t_first = norm_layer(ugf, is_train, 'UGF_%d/batch_norm' % t)
                G_t_second = norm_layer(
                    tf.nn.bias_add(
                        tf.nn.conv2d(x_t,
                                     wgr,
                                     strides=(1, 1, 1, 1),
                                     padding='VALID'), bgr), is_train,
                    'XGR_%d/batch_norm' % t)
                G_t = tf.nn.sigmoid(G_t_first + G_t_second)

                # x(t) = relu(bn(wf*u) + bn(bn(wr*x) * G))
                x_t_first = norm_layer(uf, is_train, 'UF_%d/batch_norm' % t)
                x_t_second = norm_layer(
                    tf.nn.bias_add(
                        tf.nn.conv2d(x_t,
                                     wr,
                                     strides=(1, 1, 1, 1),
                                     padding='SAME'), br), is_train,
                    'x_t_second_inner_%d/batch_norm' % t)
                x_t_second = norm_layer(x_t_second * G_t, is_train,
                                        'x_t_second_outer_%d/batch_norm' % t)
                x_t = tf.nn.relu(x_t_first + x_t_second)

        return x_t
