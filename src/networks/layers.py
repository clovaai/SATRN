"""
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import collections
import tensorflow as tf

kernel_initializer = tf.contrib.layers.xavier_initializer()
bias_initializer = tf.constant_initializer(value=0.0)

# Params
ConvParams = collections.namedtuple(
    'ConvParams', ('filters', 'kernel_size', 'strides', 'padding', 'use_bias',
                   'batch_norm', 'name'))


def residual_block(bottom,
                   conv_params,
                   is_train,
                   shortcut_conv_param=None,
                   use_shortcut_conv=False):
    """
    """
    # skip connection
    if use_shortcut_conv:
        if shortcut_conv_param is None:
            raise Exception("Short conv param is None")
        skip = conv_layer(bottom, shortcut_conv_param, is_train)

    else:
        skip = bottom

    # first layer (conv-bn-relu)
    top = conv_layer(bottom, conv_params[0], is_train)

    # second layer (conv-bn-res-relu)
    strides = (1, 1) \
        if conv_params[1].strides is None \
        else conv_params[1].strides

    top = tf.layers.conv2d(top,
                           filters=conv_params[1].filters,
                           kernel_size=conv_params[1].kernel_size,
                           strides=strides,
                           padding=conv_params[1].padding,
                           activation=None,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           name=conv_params[1].name)
    top = norm_layer(top, is_train, conv_params[1].name + '/batch_norm')

    top += skip
    top = tf.nn.relu(top, name=conv_params[1].name + '/relu')

    return top


def conv_layer(bottom,
               conv_param,
               is_train,
               use_activation=True,
               activation=tf.nn.relu):
    """
    """
    strides = (1, 1) \
        if conv_param.strides is None \
        else conv_param.strides

    if conv_param.batch_norm:
        _activation = None

    else:
        _activation = activation

    top = tf.layers.conv2d(bottom,
                           filters=conv_param.filters,
                           kernel_size=conv_param.kernel_size,
                           strides=strides,
                           padding=conv_param.padding,
                           activation=_activation,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           use_bias=conv_param.use_bias,
                           name=conv_param.name)

    if conv_param.batch_norm:
        top = norm_layer(top, is_train, conv_param.name + '/batch_norm')
        if use_activation:
            top = activation(top, name=conv_param.name + '/relu')

    return top


def pool_layer(bottom, wpool, padding, name, wstride=2):
    """
    Short function to build a pooling layer with less syntax
    """
    top = tf.layers.max_pooling2d(bottom, [2, wpool], [2, wstride],
                                  padding=padding,
                                  name=name)

    return top


def norm_layer(bottom, is_train, name):
    """
    Short function to build a batch normalization layer with less syntax
    """
    top = tf.layers.batch_normalization(bottom,
                                        axis=3,
                                        training=is_train,
                                        name=name)

    return top


def rnn_layer(bottom_sequence,
              sequence_length,
              rnn_size,
              scope,
              cell_type='lstm'):
    """
    Build bidirectional (concatenated output) RNN layer
    """
    # Default activation is tanh
    with tf.variable_scope(scope):
        if cell_type == 'lstm':
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=kernel_initializer)
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=kernel_initializer)
        elif cell_type == 'gru':
            cell_fw = tf.contrib.rnn.GRUCell(
                rnn_size, kernel_initializer=kernel_initializer)
            cell_bw = tf.contrib.rnn.GRUCell(
                rnn_size, kernel_initializer=kernel_initializer)
        else:
            raise NotImplementedError

    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope)

    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')

    return rnn_output_stack


def rnn_layers(features,
               sequence_length,
               rnn_size,
               cell_type='lstm',
               layer_count=2,
               use_projection=False):
    """
    Build a stack of RNN layers from input features
    """
    with tf.variable_scope("rnn"):
        # Transpose to time-major order for efficiency
        rnn_sequence = tf.transpose(features, perm=[1, 0, 2], name='time_major')

        for indx in range(layer_count):
            rnn_sequence = rnn_layer(rnn_sequence,
                                     sequence_length,
                                     rnn_size,
                                     'bdrnn' + str(indx + 1),
                                     cell_type=cell_type)

            if use_projection and indx != layer_count - 1:
                rnn_sequence = dense_layer(rnn_sequence, rnn_size,
                                           'rnn_hidden_fc' + str(indx + 1))

        return rnn_sequence


def dense_layer(inputs,
                units,
                name,
                activation=None,
                _kernel_initializer=None,
                _bias_initializer=None):
    """
    """
    with tf.variable_scope("dense"):
        _kernel_initializer = _kernel_initializer or kernel_initializer
        _bias_initializer = _bias_initializer or bias_initializer
        outputs = tf.layers.dense(inputs,
                                  units,
                                  activation=activation,
                                  kernel_initializer=_kernel_initializer,
                                  bias_initializer=_bias_initializer,
                                  name=name)

    return outputs


def attention_decoder(attention_states,
                      label,
                      num_classes,
                      rnn_size,
                      is_train,
                      label_maxlen=25,
                      cell_type='lstm',
                      attention_decoder_operator=None):
    """
    """
    with tf.variable_scope('attention_layer'):
        # Batch major
        attention_states = tf.transpose(attention_states,
                                        perm=[1, 0, 2],
                                        name='batch_major')

        # Make decoder inputs(dummy)
        batch_size = tf.shape(attention_states)[0]

        dummy_label = tf.concat(
            [tf.zeros([batch_size, num_classes]),
             tf.ones([batch_size, 1])],
            axis=-1)
        decoder_inputs = [dummy_label] + [None] * (label_maxlen - 1)

        # Make initial state
        if cell_type == 'lstm':
            rnn_cell = tf.contrib.rnn.LSTMCell(rnn_size)

        elif cell_type == 'gru':
            rnn_cell = tf.contrib.rnn.GRUCell(rnn_size)

        else:
            raise NotImplementedError

        initial_state = rnn_cell.zero_state(batch_size, tf.float32)

        # loop function
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
        softmax_b = tf.get_variable('softmax_b', [num_classes + 1],
                                    initializer=bias_initializer)

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

        features, _, attn_weights = attention_decoder_operator(
            decoder_inputs=decoder_inputs,
            initial_state=initial_state,
            attention_states=attention_states,
            cell=rnn_cell,
            loop_function=get_input)
        features = tf.stack(features, axis=1)
        features = tf.reshape(features, (-1, rnn_size))
        rnn_logits = tf.nn.xw_plus_b(features, softmax_w, softmax_b)
        rnn_logits = tf.reshape(rnn_logits,
                                (batch_size, label_maxlen, num_classes + 1))

        return rnn_logits, attn_weights


def _attention_decoder(decoder_inputs,
                       initial_state,
                       attention_states,
                       cell,
                       output_size=None,
                       num_heads=1,
                       loop_function=None,
                       dtype=None,
                       scope=None,
                       initial_state_attention=False):

    from tensorflow.contrib.rnn.python.ops import core_rnn_cell
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import nn_ops
    from tensorflow.python.ops import variable_scope
    from tensorflow.python.util import nest
    Linear = core_rnn_cell._Linear

    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")

    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")

    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())

    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder",
                                       dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution,
        # need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size

        for a in range(num_heads):
            k = variable_scope.get_variable(
                "AttnW_%d" % a, [1, 1, attn_size, attention_vec_size],
                dtype=dtype)
            hidden_features.append(
                nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                variable_scope.get_variable("AttnV_%d" % a,
                                            [attention_vec_size],
                                            dtype=dtype))

        state = initial_state

        def attention(query):
            """
            Put attention masks on hidden using hidden_features and query.
            """
            ds = []  # Results of attention reads will be stored here.
            ats = []

            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)

                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2

                query = array_ops.concat(query_list, 1)

            for a in range(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = Linear(query, attention_vec_size, True)(query)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    y = math_ops.cast(y, dtype)
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(math_ops.cast(s, dtype=dtypes.float32))
                    # Now calculate the attention-weighted vector d.
                    a = math_ops.cast(a, dtype)
                    ats.append(a)
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            ats = tf.stack(ats, axis=1)

            return ds, tf.reduce_max(ats, axis=1)

        outputs = []
        attn_weights = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(batch_attn_size, dtype=dtype)
            for _ in range(num_heads)
        ]

        for a in attns:
            a.set_shape([None, attn_size])

        if initial_state_attention:
            attns, ats = attention(initial_state)
            attn_weights.append(ats)

        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)

            # Merge input and previous attentions
            # into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]

            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" %
                                 inp.name)

            inputs = [inp] + attns
            inputs = [math_ops.cast(e, dtype) for e in inputs]
            x = Linear(inputs, input_size, True)(inputs)

            # Run the RNN.
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    attns, ats = attention(state)
                    attn_weights.append(ats)

            else:
                attns, ats = attention(state)
                attn_weights.append(ats)

            with variable_scope.variable_scope("AttnOutputProjection"):
                cell_output = math_ops.cast(cell_output, dtype)
                inputs = [cell_output] + attns
                output = Linear(inputs, output_size, True)(inputs)

            if loop_function is not None:
                prev = output
            outputs.append(output)

    attn_weights = tf.stack(attn_weights, axis=1)

    return outputs, state, attn_weights

