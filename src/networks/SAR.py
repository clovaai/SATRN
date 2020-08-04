"""
Re-implementaition of "Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition, Li et al."
https://arxiv.org/abs/1811.00751

Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import tensorflow as tf

from networks.layers import \
        conv_layer, pool_layer, residual_block, \
        dense_layer, ConvParams, rnn_layer
from networks.Network import Network


class SAR(Network):

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
        height = self.FLAGS.resize_hw.height
        width = self.FLAGS.resize_hw.width

        # Resize image
        image = tf.image.resize_images(image, (height, width),
                                       method=tf.image.ResizeMethod.BICUBIC)

        # Rotation augmentation
        if is_train:
            h, w, _ = image.get_shape()
            up_pad = tf.concat([image[0:1, :, :] for _ in range(2 * h)], axis=0)
            down_pad = tf.concat([image[-1:, :, :] for _ in range(2 * h)],
                                 axis=0)
            image = tf.concat([up_pad, image], axis=0)
            image = tf.concat([image, down_pad], axis=0)

            left_pad = tf.concat([image[:, 0:1, :] for _ in range(w)], axis=1)
            right_pad = tf.concat([image[:, -1:, :] for _ in range(w)], axis=1)
            image = tf.concat([left_pad, image], axis=1)
            image = tf.concat([image, right_pad], axis=1)

            theta = tf.random_normal([], mean=0.0, stddev=self.FLAGS.rot_stddev)
            image = tf.contrib.image.rotate(image,
                                            theta,
                                            interpolation='BILINEAR')
            theta = tf.abs(theta)
            h = tf.to_float(h)
            w = tf.to_float(w)
            hp = h * tf.cos(theta) + w * tf.sin(theta)
            wp = h * tf.sin(theta) + w * tf.cos(theta)
            y, x = tf.to_float(5 * h // 2), tf.to_float(3 * w // 2)

            boxes = [[(y - hp // 2) / (5 * h), (x - wp // 2) / (3 * w),
                      (y + hp // 2) / (5 * h), (x + wp // 2) / (3 * w)]]
            image = tf.expand_dims(image, 0)
            image = tf.image.crop_and_resize(image,
                                             boxes, [0], [height, width],
                                             method='bilinear')
            image = tf.squeeze(image, 0)

        return image

    def get_logits(self, image, is_train, **kwargs):
        """
        """
        # ResNet
        widths = tf.ones(tf.shape(image)[0],
                         dtype=tf.int32) * tf.shape(image)[2]
        features, sequence_length = self._convnet_layers(
            image, widths, is_train)

        # LSTM encoder
        with tf.variable_scope("rnn"):
            rnn_inputs = tf.nn.max_pool(features, (1, 8, 1, 1), (1, 1, 1, 1),
                                        'VALID',
                                        data_format='NHWC')
            rnn_inputs = tf.squeeze(rnn_inputs, axis=[1])
            rnn_inputs = tf.transpose(rnn_inputs,
                                      perm=[1, 0, 2],
                                      name='time_major')
            holistic_features = rnn_layer(rnn_inputs,
                                          sequence_length,
                                          self.rnn_size,
                                          scope='holistic')
            holistic_feature = dense_layer(holistic_features[-1],
                                           self.FLAGS.rnn_size,
                                           name='holistic_projection')

        # 2D LSTM decoder
        logits, weights = self.twodim_attention_decoder(
            holistic_feature, features, kwargs['label'], len(self.out_charset),
            self.FLAGS.rnn_size, is_train, self.FLAGS.label_maxlen)
        logits = tf.reshape(
            logits, [-1, self.FLAGS.label_maxlen,
                     len(self.out_charset) + 1])

        sequence_length = None
        self.attention_weights = tf.expand_dims(weights, axis=1)

        return logits, sequence_length

    def _convnet_layers(self, inputs, widths, is_train):
        """
        Build convolutional network layers attached to the given input tensor
        """
        conv_params = \
            [  # conv1_x
             ConvParams(64, 3, (1, 1), 'same', False, True, 'conv1_1'),
             ConvParams(128, 3, (1, 1), 'same', False, True, 'conv1_2'),
             # conv2_x
             ConvParams(256, 1, (1, 1), 'same', False, True, 'conv2_1'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'resd2_1'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'resd2_2'),
             ConvParams(256, 3, (1, 1), 'same', False, True, 'conv2_2'),
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
             ConvParams(512, 3, (1, 1), 'same', False, True, 'conv5_2')]

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

            conv4 = conv3
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
            conv5 = conv_layer(conv5, conv_params[17], is_train)

            features = conv5
            sequence_length = tf.reshape(widths // 4, [-1], name='seq_len')

            return features, sequence_length

    def twodim_attention_decoder(self,
                                 holistic_feature,
                                 attention_states,
                                 label,
                                 num_classes,
                                 rnn_size,
                                 is_train,
                                 label_maxlen=25):
        """
        """
        with tf.variable_scope('attention_layer'):
            batch_size = tf.shape(attention_states)[0]
            cell = tf.contrib.rnn.LSTMCell(rnn_size)
            dummy_label = tf.concat(
                [tf.zeros([batch_size, num_classes]),
                 tf.ones([batch_size, 1])],
                axis=-1)
            decoder_inputs = [dummy_label] + [None] * (label_maxlen - 1)

            if label is not None:
                output_shape = tf.to_int64(
                    tf.stack([batch_size, label_maxlen], axis=0))
                label = tf.sparse_to_dense(sparse_indices=label.indices,
                                           sparse_values=label.values,
                                           output_shape=output_shape,
                                           default_value=num_classes)
                label_one_hot = tf.one_hot(label, num_classes + 1)
            else:
                label_one_hot = tf.zeros([batch_size, label_maxlen])

            softmax_w = tf.get_variable(
                'softmax_w', [rnn_size, num_classes + 1],
                initializer=tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable(
                'softmax_b', [num_classes + 1],
                initializer=tf.constant_initializer(value=0.0))

            def get_train_input(prev, i):
                if i == 0:
                    return dummy_label
                else:
                    return label_one_hot[:, i - 1, :]

            def get_eval_input(prev, i):
                if i == 0:
                    return dummy_label
                else:
                    _logit = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
                    _prediction = tf.argmax(_logit, axis=-1)
                    return tf.one_hot(_prediction, num_classes + 1)

            def get_input(prev, i):
                if is_train:
                    return get_train_input(prev, i)
                else:
                    return get_eval_input(prev, i)

            # attention_states [B, 8, 25, 512]
            height = tf.shape(attention_states)[1]
            width = tf.shape(attention_states)[2]
            attn_size = rnn_size
            q = tf.get_variable("AttnQ", [1, attn_size * 2, attn_size],
                                dtype=tf.float32)
            k = tf.get_variable("AttnK", [3, 3, attn_size, attn_size],
                                dtype=tf.float32)
            v = tf.get_variable("AttnV", [1, 1, attn_size, 1], dtype=tf.float32)
            key = tf.nn.conv2d(attention_states, k, [1, 1, 1, 1], "SAME")

            def attention(query):
                with tf.variable_scope("Attention"):
                    query = tf.reshape(query, [batch_size, 1, attn_size * 2])
                    y = tf.nn.conv1d(query, q, 1, "SAME", data_format="NWC")
                    y = tf.reshape(y, [-1, 1, 1, attn_size])
                    s = tf.nn.conv2d(tf.nn.tanh(key + y), v, [1, 1, 1, 1],
                                     "SAME")
                    s = tf.reshape(s, [-1, height * width, 1])
                    a = tf.nn.softmax(s, axis=1)
                    a = tf.reshape(a, [-1, height, width, 1])
                    d = tf.reduce_sum(a * attention_states, [1, 2])

                return d, tf.reshape(a, [-1, height, width])

            attn_weights = []
            features = []
            prev = None
            state = (holistic_feature, holistic_feature)
            _state = tf.concat(state, axis=-1)
            attns, ats = attention(_state)
            attn_weights.append(ats)

            for i, inp in enumerate(decoder_inputs):

                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                if prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = get_input(prev, i)

                input_size = inp.get_shape().with_rank(2)[1]

                inputs = tf.concat([inp, attns], axis=-1)
                x = dense_layer(inputs,
                                input_size,
                                name="input_projection",
                                activation=None)

                # Run the RNN.
                cell_output, state = cell(x, state)

                # Run the attention mechanism.
                _state = tf.concat(state, axis=-1)
                attns, ats = attention(_state)
                attn_weights.append(ats)

                with tf.variable_scope("AttnOutputProjection"):
                    inputs = tf.concat([cell_output, attns], axis=-1)
                    output = dense_layer(inputs,
                                         rnn_size,
                                         name="output_projection",
                                         activation=tf.nn.relu)

                prev = output
                features.append(output)

            features = tf.stack(features, axis=1)
            features = tf.reshape(features, (-1, rnn_size))
            rnn_logits = tf.nn.xw_plus_b(features, softmax_w, softmax_b)
            rnn_logits = tf.reshape(rnn_logits,
                                    (batch_size, label_maxlen, num_classes + 1))
            attn_weights = tf.stack(attn_weights, axis=1)

            return rnn_logits, attn_weights
