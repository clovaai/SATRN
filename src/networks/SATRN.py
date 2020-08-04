"""
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import math
import tensorflow as tf

from networks.layers import \
        conv_layer, pool_layer, dense_layer, ConvParams, norm_layer
from networks.Network import Network


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    """
    """
    pass


def depthwise_conv_layer(bottom,
                         conv_param,
                         is_train,
                         use_activation=True,
                         activation=tf.nn.relu):
    """
    """
    strides = (1, 1) if conv_param.strides is None else conv_param.strides
    kernel_initializer = tf.contrib.layers.xavier_initializer()

    if conv_param.batch_norm:
        _activation = None
    else:
        _activation = activation

    top = DepthwiseConv2D([conv_param.kernel_size, conv_param.kernel_size],
                          strides=strides,
                          depthwise_initializer=kernel_initializer,
                          padding=conv_param.padding,
                          activation=_activation,
                          use_bias=False)(bottom)

    if conv_param.batch_norm:
        top = norm_layer(top, is_train, conv_param.name + '/batch_norm')
        if use_activation:
            top = activation(top, name=conv_param.name + '/relu')

    return top


class SATRN(Network):

    def __init__(self, FLAGS, out_charset):
        """
        """
        super().__init__(FLAGS, out_charset)

        # Set Default Settings
        self.loss_fn = 'cross_ent'

        self.hidden_size = FLAGS.hidden_size
        self.filter_size = FLAGS.filter_size
        self.label_maxlen = FLAGS.label_maxlen
        self.enc_layers = FLAGS.enc_layers
        self.dec_layers = FLAGS.dec_layers
        self.dropout_rate = FLAGS.dropout_rate
        self.num_heads = FLAGS.num_heads

    def resize_image(self, image, nh, nw):
        """ Resize image with piilow bicubic
        """
        from PIL import Image
        import numpy as np

        h, w, c = np.shape(image)

        if c == 1:
            img = np.reshape(image, (h, w))
            img = Image.fromarray(img, 'L')
        else:
            img = Image.fromarray(image)

        img = img.resize((nw, nh), Image.BICUBIC)
        img = np.array(img, dtype=np.uint8)
        img = np.reshape(img, (nh, nw, c))

        return img

    def preprocess_image(self, image, is_train=True):
        """
        """
        # Resize image
        height = self.FLAGS.resize_hw.height
        width = self.FLAGS.resize_hw.width
        channel = image.get_shape()[2]
        image = tf.py_func(self.resize_image, [image, height, width], tf.uint8)
        image.set_shape([height, width, channel])

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5)

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
        widths = tf.ones(tf.shape(image)[0],
                         dtype=tf.int32) * tf.shape(image)[2]
        features, shape, enc_weights = \
            self._transformer_layers(image, widths, is_train)
        batch_size = tf.shape(features)[0]

        if is_train:
            label = tf.sparse_to_dense(kwargs['label'].indices,
                                       tf.to_int64(
                                           [batch_size, self.label_maxlen]),
                                       kwargs['label'].values,
                                       default_value=len(self.out_charset))
            label = tf.reshape(label, [-1, self.label_maxlen])
            logits, weights = self.transformer_decoder(label, features,
                                                       is_train)
        else:
            logits, decoded_ids, weights = \
                    self.transformer_predictor(features,
                                               is_train,
                                               self.label_maxlen)

        logits = tf.reshape(logits,
                            [-1, self.label_maxlen,
                             len(self.out_charset) + 1])
        sequence_length = None

        return logits, sequence_length

    def _transformer_layers(self, inputs, widths, is_train):
        """
        """
        conv_params = \
            [ConvParams(self.hidden_size//2, 3,
                        (1, 1), 'same', False, True, 'conv1'),
             ConvParams(self.hidden_size, 3,
                        (1, 1), 'same', False, True, 'conv2')]

        with tf.variable_scope("transformer_layers"):
            conv1 = conv_layer(inputs, conv_params[0], is_train)
            conv1 = pool_layer(conv1, 2, 'valid', 'pool1')
            conv2 = conv_layer(conv1, conv_params[1], is_train)
            conv2 = pool_layer(conv2, 2, 'valid', 'pool2')

            features, shape, weights = \
                self.transformer_encoder(conv2, self.enc_layers,
                                         self.hidden_size, is_train)
            features = tf.reshape(features,
                                  (shape[0], shape[1] * shape[2], shape[3]))

            return features, shape, weights

    def transformer_encoder(self, features, num_layers, hidden_size, is_train):
        """
        """
        with tf.variable_scope('transformer_enc'):
            attention_bias = 0

            # Position encoding
            batch_size = tf.shape(features)[0]
            height = tf.shape(features)[1]
            width = tf.shape(features)[2]
            const_h = self.FLAGS.resize_hw.height // 4
            const_w = self.FLAGS.resize_hw.width // 4
            h_encoding = self.get_position_encoding(height, hidden_size,
                                                    'h_encoding')
            w_encoding = self.get_position_encoding(width, hidden_size,
                                                    'w_encoding')
            h_encoding = tf.expand_dims(h_encoding, axis=1)
            w_encoding = tf.expand_dims(w_encoding, axis=0)
            h_encoding = tf.tile(tf.expand_dims(h_encoding, axis=0),
                                 [batch_size, 1, 1, 1])
            w_encoding = tf.tile(tf.expand_dims(w_encoding, axis=0),
                                 [batch_size, 1, 1, 1])

            # Adaptive 2D potisiontal encoding
            inter = tf.reduce_mean(features, axis=[1, 2])  # [B, hidden]
            inter = dense_layer(inter,
                                hidden_size // 2,
                                name='intermediate',
                                activation=tf.nn.relu)

            if is_train:
                inter = tf.nn.dropout(inter, self.dropout_rate)

            alpha = dense_layer(inter,
                                2 * hidden_size,
                                name='alpha',
                                activation=tf.nn.sigmoid)
            alpha = tf.reshape(alpha, [-1, 2, 1, hidden_size])
            pos_encoding = alpha[:, 0:1, :, :] * h_encoding \
                + alpha[:, 1:2, :, :] * w_encoding

            features += pos_encoding
            self.hw = tf.reduce_sum(alpha, axis=[2, 3])

            # Save shape
            shape = (-1, height, width, hidden_size)
            features = tf.reshape(features, (-1, height * width, hidden_size))

            # Dropout
            if is_train:
                features = tf.nn.dropout(features, self.dropout_rate)

            # Encoder stack
            ws = []
            for n in range(num_layers):
                with tf.variable_scope("encoder_layer_%d" % n):
                    with tf.variable_scope("self_attention"):
                        # layer norm
                        y = self.layer_norm(features, hidden_size)

                        # self att
                        y, w = self.attention_layer(y, y, hidden_size,
                                                    attention_bias, 'self_att',
                                                    is_train)
                        ws.append(w)

                        # dropout
                        if is_train:
                            y = tf.nn.dropout(y, self.dropout_rate)

                        # skip
                        features = y + features

                    with tf.variable_scope("ffn"):
                        # layer norm
                        y = self.layer_norm(features, hidden_size)

                        # cnn
                        y = tf.reshape(features, shape)

                        conv_params = \
                            [ConvParams(self.filter_size, 1, (1, 1),
                                        'same', False, True, 'expand'),
                             ConvParams(self.filter_size, 3, (1, 1),
                                        'same', False, True, 'dwconv'),
                             ConvParams(self.hidden_size, 1, (1, 1),
                                        'same', False, True, 'reduce')]
                        y = conv_layer(y, conv_params[0], is_train)
                        y = depthwise_conv_layer(y, conv_params[1], is_train)
                        y = conv_layer(y, conv_params[2], is_train)
                        y = tf.reshape(y, (-1, height * width, hidden_size))

                        # skip
                        features = y + features

            # Output normalization
            features = self.layer_norm(features, hidden_size)
            ws = tf.stack(ws, axis=1)

        return features, shape, ws

    def transformer_decoder(self, labels, encoder_outputs, is_train):
        """
        """
        with tf.variable_scope('transformer_dec'):
            batch_size = tf.shape(labels)[0]

            # Shift targets to the right, and remove the last element
            with tf.name_scope("shift_targets"):
                init_ids = tf.ones([batch_size, 1], dtype=tf.int32) * len(
                    self.out_charset)
                labels = tf.concat([init_ids, labels], axis=1)[:, :-1]

            # Input embedding
            decoder_inputs = self.embedding_softmax_layer(labels)

            # Position encoding
            length = tf.shape(decoder_inputs)[1]
            pos_encoding = self.get_position_encoding(length, self.hidden_size,
                                                      'dec_pos')
            decoder_inputs += pos_encoding

            # Dropout
            if is_train:
                decoder_inputs = tf.nn.dropout(decoder_inputs,
                                               self.dropout_rate)

            # Attention bias
            self_attention_bias = self.get_decoder_self_attention_bias(length)
            attention_bias = 0

            decoder_outputs, weights = self.decoder_stack(
                decoder_inputs, encoder_outputs, self_attention_bias,
                attention_bias, is_train, None)
            logits = self.embedding_projection(decoder_outputs)

        return logits, weights

    def transformer_predictor(self, encoder_outputs, is_train, label_maxlen=25):
        """
        """
        with tf.variable_scope('transformer_dec', reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(encoder_outputs)[0]
            pos_encoding = self.get_position_encoding(label_maxlen + 1,
                                                      self.hidden_size,
                                                      'dec_pos')

            # Create initial set of IDs that will be passed
            # into symbols_to_logits_fn.
            ids = tf.ones([batch_size, 1], dtype=tf.int32) * len(
                self.out_charset)

            # Attention bias
            self_attention_bias = \
                self.get_decoder_self_attention_bias(label_maxlen)
            attention_bias = 0

            decoded_ids = []
            logits = []

            # Create cache storing decoder attention values for each layer.
            cache = {
                "layer_%d" % layer: {
                    "k": tf.zeros([batch_size, 0, self.hidden_size // 4]),
                    "v": tf.zeros([batch_size, 0, self.hidden_size]),
                } for layer in range(self.dec_layers)
            }
            weights = []

            for i in range(label_maxlen):
                decoder_inputs = self.embedding_softmax_layer(ids)
                decoder_inputs += pos_encoding[i:i + 1]

                decoder_outputs, ws = self.decoder_stack(
                    decoder_inputs, encoder_outputs,
                    self_attention_bias[:, :, i:i + 1, :i + 1], attention_bias,
                    is_train, cache)
                weights.append(ws)

                logit = self.embedding_projection(decoder_outputs)
                ids = tf.argmax(logit, axis=-1)
                decoded_ids.append(ids)
                logits.append(logit)
            logits = tf.concat(logits, axis=1)
            weights = tf.concat(weights, axis=3)

            return logits, decoded_ids, weights

    def decoder_stack(self,
                      decoder_inputs,
                      encoder_outputs,
                      self_attention_bias,
                      attention_bias,
                      is_train,
                      cache=None):
        """
        """
        ws = []

        # Decoder stack
        for n in range(self.dec_layers):
            with tf.variable_scope("decoder_layer_%d" % n):
                layer_name = "layer_%d" % n
                layer_cache = cache[layer_name] if cache is not None else None

                with tf.variable_scope("self_attention"):
                    # layer norm
                    y = self.layer_norm(decoder_inputs, self.hidden_size)

                    # self att
                    y, _ = self.attention_layer(y, y, self.hidden_size,
                                                self_attention_bias, 'self_att',
                                                is_train, layer_cache)

                    # dropout
                    if is_train:
                        y = tf.nn.dropout(y, self.dropout_rate)

                    # skip
                    decoder_inputs = y + decoder_inputs

                with tf.variable_scope("encdec_attention"):
                    # layer norm
                    y = self.layer_norm(decoder_inputs, self.hidden_size)

                    # self att
                    y, w = self.attention_layer(y, encoder_outputs,
                                                self.hidden_size,
                                                attention_bias, 'encdec_att',
                                                is_train)
                    ws.append(w)

                    # dropout
                    if is_train:
                        y = tf.nn.dropout(y, self.dropout_rate)

                    # skip
                    decoder_inputs = y + decoder_inputs

                with tf.variable_scope("ffn"):
                    # layer norm
                    y = self.layer_norm(decoder_inputs, self.hidden_size)

                    # ffn
                    y = dense_layer(y,
                                    self.filter_size,
                                    name='filter_layer',
                                    activation=tf.nn.relu)

                    # dropout
                    if is_train:
                        y = tf.nn.dropout(y, self.dropout_rate)

                    y = dense_layer(y,
                                    self.hidden_size,
                                    name='output_layer',
                                    activation=tf.nn.relu)
                    # dropout
                    if is_train:
                        y = tf.nn.dropout(y, self.dropout_rate)

                    # skip
                    decoder_inputs = y + decoder_inputs

        # Output normalization
        decoder_outputs = self.layer_norm(decoder_inputs, self.hidden_size)
        ws = tf.stack(ws, axis=1)

        return decoder_outputs, ws

    def attention_layer(self,
                        x,
                        y,
                        hidden_size,
                        bias,
                        name,
                        is_train,
                        cache=None):
        """
        """
        # Query Key Value projection
        q = dense_layer(x, hidden_size // 4, name='q')
        k = dense_layer(y, hidden_size // 4, name='k')
        v = dense_layer(y, hidden_size, name='v')

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split head (for multi head attention)
        q = self.split_heads(q, hidden_size // 4)
        k = self.split_heads(k, hidden_size // 4)
        v = self.split_heads(v, hidden_size)

        # Scale q to prevent the dot product
        # between q and k from growing too large.
        depth = (hidden_size // self.num_heads)
        q *= depth**-0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        w = tf.nn.softmax(logits, name="attention_weights")

        if is_train:
            w = tf.nn.dropout(w, self.dropout_rate)

        attention_output = tf.matmul(w, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output, hidden_size)

        # Run the combined outputs through another linear projection layer.
        attention_output = dense_layer(attention_output,
                                       hidden_size,
                                       name='att_out')

        return attention_output, w

    def combine_heads(self, x, hidden_size):
        """
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, hidden_size])

    def split_heads(self, x, hidden_size):
        """
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def layer_norm(self, x, hidden_size, epsilon=1e-6):
        """
        """
        scale = tf.get_variable("layer_norm_scale", [hidden_size],
                                initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [hidden_size],
                               initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

        return norm_x * scale + bias

    def get_position_encoding(self,
                              length,
                              hidden_size,
                              name,
                              min_timescale=1.0,
                              max_timescale=1.0e4):
        """
        """
        position = tf.to_float(tf.range(length))
        num_timescales = hidden_size // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
        inv_timescales = \
            min_timescale * \
            tf.exp(tf.to_float(tf.range(num_timescales))
                   * -log_timescale_increment)

        scaled_time = tf.expand_dims(position, 1) * \
            tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

        return signal

    def get_decoder_self_attention_bias(self, length):
        """
        """
        with tf.name_scope("decoder_self_attention_bias"):
            valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
            valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
            decoder_bias = -1e9 * (1.0 - valid_locs)

        return decoder_bias

    def embedding_softmax_layer(self, x):
        """
        """
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            num_classes = len(self.out_charset)
            mask = tf.to_float(tf.not_equal(x, num_classes))
            shared_weights = tf.get_variable(
                "weights", [num_classes + 1, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    0., self.hidden_size**-0.5))
            embeddings = tf.gather(shared_weights, x)
            embeddings *= tf.expand_dims(mask, -1)
            embeddings *= self.hidden_size**0.5

            return embeddings

    def embedding_projection(self, x):
        """
        """
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            num_classes = len(self.out_charset)
            shared_weights = tf.get_variable(
                "weights", [num_classes + 1, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    0., self.hidden_size**-0.5))
            length = tf.shape(x)[1]
            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, shared_weights, transpose_b=True)

            return tf.reshape(logits, [-1, length, num_classes + 1])
