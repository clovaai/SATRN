"""
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
import re
import importlib
import collections
import pickle as cp
import tensorflow as tf
import unicodedata

from constant import SPE_TOKENS, UNK_INDEX, EOS_INDEX, DELIMITER, EOS_TOKEN

TowerResult = collections.namedtuple(
    'TowerResult', ('tvars', 'loss', 'grads', 'extra_update_ops', 'error',
                    'prediction', 'text', 'filename', 'dataset'))


def _get_init_pretrained(tune_from):
    """ Returns lambda for reading pretrained initial model.
    """
    if not tune_from:
        return None

    saver_reader = tf.train.Saver()
    model_path = tune_from

    def init_fn(scaffold, sess):
        return saver_reader.restore(sess, model_path)

    return init_fn


def get_scaffold(saver, tune_from, name):
    """ Get scaffold containing initializer operation for MonitoredTrainingSession.
    """
    if name == 'train':
        iterator_init = [
            tf.get_collection('TRAIN_ITERATOR'),
            tf.get_collection('VALID_ITERATOR')
        ]
    elif name == 'eval':
        iterator_init = [tf.get_collection('EVAL_ITERATOR')]
    else:
        raise NotImplementedError

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge([s for s in summaries if name in s.name])
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
    )
    local_init_op = tf.group(tf.tables_initializer(),
                             tf.local_variables_initializer(), *iterator_init)
    scaffold = tf.train.Scaffold(saver=saver,
                                 init_op=init_op,
                                 local_init_op=local_init_op,
                                 summary_op=summary_op,
                                 init_fn=_get_init_pretrained(tune_from))
    return scaffold


def get_session(monitored_sess):
    """ Get Session object from MonitoredTrainingSession.
    """
    session = monitored_sess
    while type(session).__name__ != 'Session':
        session = session._sess

    return session


def validate(sess,
             step,
             val_tower_outputs,
             out_charset,
             is_ctc,
             val_summary_op=None,
             val_summary_writer=None,
             val_saver=None,
             best_val_err_rates=None,
             best_steps=None,
             best_model_dir=None,
             lowercase=False,
             alphanumeric=False):
    """ Run validation.
    """
    val_cnts = {'total_valid': 0}
    val_errs = {'total_valid': 0}
    val_err_rates = {}
    val_preds = {}
    val_sums = {}

    while True:
        try:
            if val_summary_op is not None:
                val_results, val_summary = sess.run(
                    [val_tower_outputs, val_summary_op])
            else:
                val_results = sess.run(val_tower_outputs)
                val_summary = None

            for val_result in val_results:
                _, preds, gts, filenames, datasets = val_result

                for dataset, filename, p, g in \
                        zip(datasets, filenames, preds, gts):

                    s = get_string(p, out_charset, is_ctc=is_ctc)
                    g = g.decode('utf8').replace(DELIMITER, '')
                    dataset = dataset.decode('utf8')

                    s = adjust_string(s, lowercase, alphanumeric)
                    g = adjust_string(g, lowercase, alphanumeric)
                    e = int(s != g)

                    if dataset in val_cnts.keys():
                        val_cnts[dataset] += 1
                        val_errs[dataset] += e
                        val_preds[dataset].append((filename, s, g))

                    else:
                        val_cnts[dataset] = 1
                        val_errs[dataset] = e
                        val_preds[dataset] = [(filename, s, g)]

                    val_cnts['total_valid'] += 1
                    val_errs['total_valid'] += e

            val_sums[dataset] = val_summary

        except tf.errors.OutOfRangeError:

            # Write summary
            for dataset in val_cnts.keys():
                val_cnt = val_cnts[dataset]
                val_err = val_errs[dataset]
                val_err_rate = float(val_err) / val_cnt
                val_err_rates[dataset] = val_err_rate

                if val_summary_writer is not None:

                    val_err_summary = tf.Summary()
                    val_err_summary.value.add(tag='valid/sequence_error',
                                              simple_value=val_err_rate)
                    val_summary_writer[dataset].add_summary(
                        val_err_summary, step)

                    if dataset != 'total_valid':
                        val_summary_writer[dataset].add_summary(
                            val_sums[dataset], step)

                if val_saver is not None:
                    if dataset not in best_val_err_rates.keys() or \
                            val_err_rate < best_val_err_rates[dataset]:

                        best_val_err_rates[dataset] = val_err_rate

                        val_saver.save(get_session(sess),
                                       os.path.join(best_model_dir, dataset))
                        best_steps[dataset] = step

            # Reset iterator
            sess.run(tf.get_collection('VALID_ITERATOR'))
            break

    return val_cnts, val_errs, val_err_rates, val_preds


def single_tower(net, gpu_indx, dataset_loader, out_charset, optimizer, name,
                 is_train):
    """ Single model tower.
    """
    # Get batch
    with tf.device('/gpu:%d' % gpu_indx), \
            tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), \
            tf.name_scope(name):

        batch = dataset_loader.get_batch()

        # Tensorboard image
        tf.summary.image('image', batch.image, max_outputs=10)

        # Inference
        logits, sequence_length = net.get_logits(
            batch.image, is_train, label=batch.label if is_train else None)

        tvars, loss, extra_update_ops = net.get_loss(
            logits,
            batch.label,
            sequence_length=sequence_length,
            label_length=batch.length,
            label_maxlen=dataset_loader.label_maxlen)

        grads = optimizer.compute_gradients(
                loss, tvars,
                colocate_gradients_with_ops=True) \
            if optimizer is not None \
            else None

        prediction, log_prob = net.get_prediction(logits, sequence_length)

        error = get_error(prediction,
                          batch.label,
                          batch.length,
                          out_charset,
                          name='err',
                          max_text_summary=10)

        prediction = tf.sparse_to_dense(sparse_indices=prediction.indices,
                                        sparse_values=prediction.values,
                                        output_shape=prediction.dense_shape,
                                        default_value=len(out_charset))

        result = TowerResult(tvars, loss, grads, extra_update_ops, error,
                             prediction, batch.text, batch.filename,
                             batch.dataset_name)

    return result


def count_available_gpus():
    """ Retrun the number of available gpus.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_cnt = len(
        [x.name for x in local_device_protos if x.device_type == 'GPU'])

    return gpu_cnt


def get_session_config():
    """ Setup session config to soften device placement.
    """
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)

    return config


def get_init_trained():
    """ Return init function to restore trained model from a given checkpoint.
    """
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    def _init_fn(sess, model_path):
        saver_reader.restore(sess, model_path)
        print('[+] Restored from {}'.format(model_path))

    return _init_fn


def get_string(labels, out_charset, is_ctc=False):
    """ Transform an 1D array of labels into the corresponding character string.
    """
    if is_ctc:
        string = ''.join(
            [out_charset[c] for c in labels if c < len(out_charset)])

    else:
        string = ''.join([
            out_charset[c] if c < len(out_charset) else '<SPE>' for c in labels
        ])
        string = string[:string.find(EOS_TOKEN) + len(EOS_TOKEN)]

    return string


def get_labels(text, out_charset):
    """ Transform text to sequence of integer.
    """
    labels = [
        out_charset.index(c) if c in list(out_charset) else UNK_INDEX
        for c in list(text)
    ] + [EOS_INDEX]

    return labels


def get_error(prediction,
              label,
              label_length,
              out_charset,
              name,
              max_text_summary=10):
    """ Returns the wrong number between prediction and label.
    """
    # Calculate Edit dist.
    hypothesis = tf.cast(prediction, tf.int32)
    edit_distance = tf.edit_distance(hypothesis, label, normalize=False)

    # Count wrong seq.
    error_count = tf.count_nonzero(edit_distance, axis=0)

    # Summary predicted text
    num_classes = len(out_charset)
    prediction = tf.sparse_tensor_to_dense(prediction,
                                           default_value=num_classes)
    label = tf.sparse_tensor_to_dense(label, default_value=num_classes)
    table = get_itos_table(out_charset)
    pr_text = tf.reduce_join(table.lookup(tf.to_int64(prediction)),
                             1,
                             keep_dims=True)
    la_text = tf.reduce_join(table.lookup(tf.to_int64(label)),
                             1,
                             keep_dims=True)
    tf.summary.text('label', la_text[:max_text_summary])
    tf.summary.text('%s/predicted' % name, pr_text[:max_text_summary])

    return error_count


def get_itos_table(out_charset):
    """ Get integer to character mapping table.
    """
    with tf.device('/cpu:0'):
        mapping_strings = tf.constant(list(out_charset))
        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping=mapping_strings, default_value='')

    return table


def get_network(FLAGS, out_charset):
    """ Get text recognition network object.
    """
    net_cls = getattr(
        importlib.import_module('networks.{}'.format(FLAGS.network)),
        FLAGS.network)
    net = net_cls(FLAGS, out_charset)

    return net


def load_charset(charset_path):
    """ Load character set.
    """
    if os.path.exists(charset_path):
        _, ext = os.path.splitext(charset_path)

        if ext == '.pkl' or ext == '.cp':
            charset = cp.load(open(charset_path, 'rb'))
        else:
            charset = open(charset_path, 'r', encoding='utf8').readline()

    elif charset_path == 'alphanumeric':
        charset = \
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    elif charset_path == 'alphanumeric_lower':
        charset = "abcdefghijklmnopqrstuvwxyz0123456789"

    else:
        raise NotImplementedError

    out_charset = add_special_symbol_to_charset(charset)

    return out_charset


def add_special_symbol_to_charset(charset):
    """ Add special symbol to character set.
    """
    out_charset = SPE_TOKENS + list(charset)

    return out_charset


def cyclic_learning_rate(global_step,
                         learning_rate,
                         max_lr=0.1,
                         lr_divider=10,
                         cut_point=10,
                         step_size=20.,
                         gamma=0.99994,
                         mode='triangular',
                         name=None):
    """ Cyclic learning rate
        https://arxiv.org/abs/1506.01186
    """
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.eager import context

    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")

    with ops.name_scope(name, "CyclicLearningRate",
                        [learning_rate, global_step]) as name:
        step_size += 1
        cycle_step = int(step_size * (1 - cut_point / 100) / 2)
        learning_rate = ops.convert_to_tensor(learning_rate,
                                              name="learning_rate")
        dtype = learning_rate.dtype
        _global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)
        iteration = tf.mod(_global_step, step_size)

        def cyclic_lr():

            def _end_fn():
                cut = (iteration - 2 * cycle_step) / (step_size -
                                                      2 * cycle_step)
                lr = max_lr * (1 + (cut * (1 - 100) / 100)) / lr_divider
                return math_ops.cast(lr, dtype)

            def _greater_fn():
                cut = 1 - (iteration - cycle_step) / cycle_step
                lr = max_lr * (1 + cut * (lr_divider - 1)) / lr_divider
                return math_ops.cast(lr, dtype)

            def _less_fn():
                cut = iteration / cycle_step
                lr = max_lr * (1 + cut * (lr_divider - 1)) / lr_divider
                return math_ops.cast(lr, dtype)

            lr = tf.cond(
                tf.greater(iteration, 2 * cycle_step), _end_fn, lambda: tf.cond(
                    tf.greater(iteration, cycle_step), _greater_fn, _less_fn))
            return lr

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()

        return cyclic_lr


def get_optimizer(optimizer_config, global_step):
    """ Get optimizer object.
    """
    learning_rate = tf.Variable(optimizer_config.learning_rate)
    momentum = tf.Variable(0.9, name='momentum')

    # Cyclic lr or LR Finder
    if optimizer_config.use_clr:
        learning_rate = cyclic_learning_rate(
            global_step,
            0.,
            max_lr=optimizer_config.learning_rate,
            step_size=optimizer_config.clr_step)

    # Optimizer
    if optimizer_config.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=momentum,
                                           beta2=optimizer_config.beta2)

    elif optimizer_config.optimizer == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

    elif optimizer_config.optimizer == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                              momentum=momentum)

    elif optimizer_config.optimizer == 'GradDesc':
        optimizer = \
                tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    else:
        raise NotImplementedError

    tf.summary.scalar('train/learning_rate', learning_rate)

    return optimizer, learning_rate


def adjust_string(s, lowercase, alphanumeric):
    """ Adjust string.
    """
    if s.find(EOS_TOKEN) > 0:
        s = s[:s.find(EOS_TOKEN)]

    if alphanumeric:
        s = ''.join([c for c in list(s) if re.match('^[a-zA-Z0-9]*$', c)])

    if lowercase:
        s = s.lower()

    return s


def text_length(text):
    """ Get text length
    """
    text = text.decode('utf-8')
    return len(text)
