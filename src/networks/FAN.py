"""
Re-implementaition of "Focusing Attention: Towards Accurate Text Recognition in Natural Images, Cheng et al."
https://arxiv.org/abs/1709.02054

Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import tensorflow as tf

from networks.layers import \
        rnn_layer, attention_decoder, conv_layer,\
        pool_layer, residual_block, ConvParams
from networks.Network import Network


class FAN(Network):

    def __init__(self, FLAGS, out_charset):
        """
        """
        super().__init__(FLAGS, out_charset)

        # Set Default Settings
        self.loss_fn = 'cross_ent'
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
        features = tf.transpose(features, perm=[1, 0, 2], name='time_major')
        attention_states = rnn_layer(features,
                                     sequence_length,
                                     self.rnn_size,
                                     scope="rnn")
        logits, weights = attention_decoder(attention_states, kwargs['label'],
                                            len(self.out_charset),
                                            self.rnn_size, is_train,
                                            self.FLAGS.label_maxlen)

        return logits, sequence_length

    def _convnet_layers(self, inputs, widths, is_train):
        """
        Build convolutional network layers attached to the given input tensor
        """
        conv_params = \
            [  # conv1_x
             ConvParams(32, 3, (1, 1), 'same', False, True, 'conv1_1'),
             ConvParams(64, 3, (1, 1), 'same', False, True, 'conv1_2'),
             # conv2_x
             ConvParams(128, 1, (1, 1), 'same', False, True, 'conv2_1'),
             ConvParams(128, 3, (1, 1), 'same', False, True, 'resd2_1'),
             ConvParams(128, 3, (1, 1), 'same', False, True, 'resd2_2'),
             ConvParams(128, 3, (1, 1), 'same', False, True, 'conv2_2'),
             # conv3_x
             ConvParams(256, 1, (1, 1), 'same', False, True, 'conv3_1'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'resd3_1'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'resd3_2'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'conv3_2'),
             # conv4_x
             ConvParams(512, 1, (1, 1), 'same', False, True, 'conv4_1'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'resd4_1'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'resd4_2'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv4_2'),
             # conv5_x
             ConvParams(512, 1, (1, 1), 'same', False, True, 'conv5_1'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'resd5_1'),
             ConvParams(512, 3, (1, 1), 'same', False, True, 'resd5_2'),
             ConvParams(512, 2, (2, 1), 'valid', False, True, 'conv5_2'),
             ConvParams(512, 2, (1, 1), 'valid', False, True, 'conv5_3')]

        with tf.variable_scope("convnet"):
            conv1 = conv_layer(inputs, conv_params[0], is_train)
            conv1 = conv_layer(conv1, conv_params[1], is_train)

            conv2 = pool_layer(conv1, 2, 'valid', 'pool2')
            conv2 = residual_block(conv2,
                                   conv_params[3:5],
                                   is_train,
                                   shortcut_conv_param=conv_params[2],
                                   use_shortcut_conv=True)
            conv2 = conv_layer(conv2, conv_params[5], is_train)

            conv3 = pool_layer(conv2, 2, 'valid', 'pool3')

            for i in range(2):
                with tf.variable_scope('conv3_{}'.format(i)):
                    conv3 = residual_block(
                        conv3,
                        conv_params[7:9],
                        is_train,
                        shortcut_conv_param=(conv_params[6]
                                             if i == 0 else None),
                        use_shortcut_conv=(i == 0))
            conv3 = conv_layer(conv3, conv_params[9], is_train)

            conv4 = tf.pad(conv3, [[0, 0], [0, 0], [1, 1], [0, 0]])
            conv4 = pool_layer(conv4, 2, 'valid', 'pool4', wstride=1)

            for i in range(5):
                with tf.variable_scope('conv4_{}'.format(i)):
                    conv4 = residual_block(
                        conv4,
                        conv_params[11:13],
                        is_train,
                        shortcut_conv_param=(conv_params[10]
                                             if i == 0 else None),
                        use_shortcut_conv=(i == 0))
            conv4 = conv_layer(conv4, conv_params[13], is_train)

            conv5 = conv4

            for i in range(3):
                with tf.variable_scope('conv5_{}'.format(i)):
                    conv5 = residual_block(
                        conv5,
                        conv_params[15:17],
                        is_train,
                        shortcut_conv_param=(conv_params[14]
                                             if i == 0 else None),
                        use_shortcut_conv=(i == 0))
            conv5 = tf.pad(conv5, [[0, 0], [0, 0], [1, 1], [0, 0]])
            conv5 = conv_layer(conv5, conv_params[17], is_train)
            conv5 = conv_layer(conv5, conv_params[18], is_train)

            features = tf.squeeze(conv5, axis=1, name='features')

            sequence_length = widths // 4 + 1
            sequence_length = tf.reshape(sequence_length, [-1], name='seq_len')

            return features, sequence_length
