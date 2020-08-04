"""
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import abc
import tensorflow as tf
import numpy as np
from constant import EOS_INDEX


class Network(abc.ABC):

    def __init__(self, FLAGS, out_charset):
        """
        """
        self.valid_loss_fn = ['ctc_loss', 'cross_ent']
        self.FLAGS = FLAGS
        self.out_charset = out_charset

    def preprocess_image_with_seed(self, image, seed, is_train=True):
        """
        """
        tf.random.set_random_seed(seed)

        return self.preprocess_image(image, is_train)

    def preprocess_image(self, image, is_train=True):
        """
        """
        return image

    @abc.abstractmethod
    def get_logits(self, image, is_train, **kwargs):
        """
        """
        return

    def get_loss(self, logits, label, **kwargs):
        """
        """
        if self.loss_fn == 'cross_ent':
            tvars, loss, extra_update_ops = \
                    self._get_cross_entropy(logits,
                                            label,
                                            kwargs['label_length'],
                                            kwargs['label_maxlen'])

        elif self.loss_fn == 'ctc_loss':
            tvars, loss, extra_update_ops = \
                    self._get_ctc_loss(logits,
                                       label,
                                       kwargs['sequence_length'])

        else:
            raise NotImplementedError

        tf.summary.scalar('loss', loss)

        return tvars, loss, extra_update_ops

    def get_prediction(self, logits, sequence_length=None):
        """
        """
        if self.loss_fn == 'cross_ent':
            return self._get_argmax_prediction(logits)

        elif self.loss_fn == 'ctc_loss':
            return self._get_ctc_prediction(logits, sequence_length)

        else:
            raise NotImplementedError

        return

    def _get_cross_entropy(self, logits, label, label_length, label_maxlen=25):
        """
        """
        with tf.name_scope("train"):
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            # Compute sequence loss
            batch_size = tf.shape(logits)[0]
            output_shape = tf.to_int64(
                tf.stack([batch_size, label_maxlen], axis=0))
            num_classes = len(self.out_charset)
            label = tf.sparse_to_dense(sparse_indices=label.indices,
                                       sparse_values=label.values,
                                       output_shape=output_shape,
                                       default_value=num_classes)
            loss = 0

            for i in range(label_maxlen):
                _logit = logits[:, i, :]
                _label = label[:, i]
                _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=_logit, labels=_label)
                _mask = tf.to_float(tf.greater_equal(label_length, i))
                _mask = tf.reshape(_mask, tf.shape(_loss))
                loss += tf.reduce_sum(_loss * _mask)

            loss /= tf.to_float(batch_size)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        return tvars, loss, extra_update_ops

    def _get_ctc_loss(self, logits, label, sequence_length):
        """
        """
        with tf.name_scope("train"):
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            loss = tf.nn.ctc_loss(label,
                                  logits,
                                  sequence_length,
                                  time_major=True)
            loss = tf.reduce_mean(loss)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        return tvars, loss, extra_update_ops

    def _get_argmax_prediction(self, logits):
        """
        """
        num_classes = len(self.out_charset)
        _, label_maxlen, _ = logits.get_shape()
        batch_size = tf.shape(logits)[0]
        predictions = []
        is_valid = tf.ones([batch_size], dtype=tf.int64)
        log_prob = tf.zeros([batch_size], dtype=tf.float32)
        valid_length = tf.zeros([batch_size], dtype=tf.float32)

        for i in range(label_maxlen):
            _logit = logits[:, i, :]
            _prob = tf.nn.softmax(_logit, axis=-1)
            _pred = tf.argmax(_logit, axis=-1)
            _prob = tf.reduce_max(_prob, axis=-1)
            _pred = _pred * is_valid + num_classes * (1 - is_valid)
            predictions.append(_pred)
            log_prob += tf.log(_prob) * tf.to_float(is_valid)
            valid_length += tf.to_float(is_valid)
            is_valid *= tf.to_int64(tf.not_equal(_pred, EOS_INDEX))

        predictions = tf.stack(predictions, axis=1)
        predictions = tf.contrib.layers.dense_to_sparse(predictions,
                                                        eos_token=num_classes)

        log_prob /= (valid_length + 1e-9)
        prob = tf.exp(log_prob)

        return predictions, prob

    def _get_ctc_prediction(self, logits, sequence_length):
        """
        """
        predictions, log_prob = tf.nn.ctc_beam_search_decoder(
            logits,
            sequence_length,
            beam_width=1,
            top_paths=1,
            merge_repeated=False)

        prob = tf.exp(log_prob)

        return predictions[0], prob
