"""
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
import sys
import glob
import copy
import collections

import tensorflow as tf
from constant import DELIMITER, UNK_INDEX, EOS_TOKEN
from utils import text_length

Batch = collections.namedtuple(
    'Batch', ('image', 'label', 'length', 'text', 'filename', 'dataset_name'))


def get_stoi_table(out_charset):
    """ Get character to integer mapping table.
    """
    with tf.device('/cpu:0'):
        mapping_strings = tf.constant(list(out_charset))
        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=mapping_strings, num_oov_buckets=0, default_value=UNK_INDEX)

    return table


class DatasetLodaer(object):
    """
    """

    def __init__(self,
                 dataset_paths,
                 dataset_portions,
                 batch_size,
                 label_maxlen,
                 out_charset,
                 preprocess_image,
                 is_train,
                 is_ctc,
                 shuffle_and_repeat,
                 concat_batch,
                 input_device,
                 num_cpus,
                 num_gpus=None,
                 worker_index=None,
                 use_rgb=False,
                 seed=None,
                 name='train'):

        self.dataset_paths = copy.deepcopy(dataset_paths)
        self.dataset_portions = dataset_portions

        if type(self.dataset_paths) is str:
            self.dataset_paths = [self.dataset_paths]

        if type(self.dataset_portions) is float:
            self.dataset_portions = [self.dataset_portions]

        if self.dataset_portions is not None:
            assert -1e-9 < abs(1. - sum(self.dataset_portions)) < 1e-9, \
                    str(sum(self.dataset_portions))
            assert len(self.dataset_paths) == len(self.dataset_portions)
        else:
            assert not concat_batch, \
                    'Data portions must be specified for concat_batch'
            self.dataset_portions = [1 for _ in range(len(self.dataset_paths))]

        _, ext = os.path.splitext(
            os.path.basename(
                glob.glob(self.dataset_paths[0], recursive=True)[0]))

        self.ext = ext
        self.tmppaths = []

        if ext == '.tfrecord':
            self.dataset_class = tf.data.TFRecordDataset
            self.parse_fn = self.tfrecord_parse_fn

        elif ext == '.txt':
            self.flatten_datasets()
            self.dataset_class = tf.data.TextLineDataset
            self.parse_fn = self.txt_parse_fn

        else:
            raise NotImplementedError

        self.dataset_names = self.get_dataset_names(self.dataset_paths)

        self.batch_size = batch_size
        self.label_maxlen = label_maxlen
        self.out_charset = out_charset
        self.table = get_stoi_table(out_charset)
        self.preprocess_image = preprocess_image \
            if preprocess_image is not None \
            else self._default_preprocess_image
        self.is_train = is_train

        assert name.lower() in ['train', 'valid', 'eval']
        self.iterator_name = name.upper() + '_ITERATOR'
        self.exclude_eos = is_ctc
        self.shuffle_and_repeat = shuffle_and_repeat
        self.concat_batch = concat_batch
        self.input_device = input_device
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.worker_index = worker_index
        self.img_channels = 3 if use_rgb else 1
        self.seed = seed
        self.buffer_size = 15000 // self.num_gpus

    def flatten_datasets(self):
        """
        """
        tmpdir = 'tmp'
        os.makedirs(tmpdir, exist_ok=True)

        for i, path in enumerate(self.dataset_paths):

            tmppath = '__'.join(os.path.join(path, 'gt.txt').split('/'))
            tmppath = os.path.join(tmpdir, tmppath)

            if not os.path.exists(tmppath):
                paths = glob.glob(path, recursive=True)
                lines = []

                for p in paths:
                    dirname = os.path.abspath(os.path.dirname(p))
                    _lines = []

                    for line in open(p, 'r', encoding='utf-8').readlines():
                        line = line.strip()

                        if line.find('\t') > 0:
                            f, g = line.split('\t', 1)
                        else:
                            f, g = line, ''

                        _lines.append(os.path.join(dirname, f) + '\t' + g)

                    lines.extend(_lines)

                tmpfile = open(tmppath, 'w', encoding='utf-8')

                for line in lines:
                    tmpfile.write(line + '\n')

                tmpfile.close()

            self.dataset_paths[i] = tmppath
            self.tmppaths.append(tmppath)

        return

    def flush_tmpfile(self):
        """
        """
        if not self.tmppaths:
            return

        tmpdir = os.path.dirname(self.tmppaths[0])

        for tmppath in self.tmppaths:
            if os.path.exists(tmppath):
                os.remove(tmppath)

        os.system('rmdir {}'.format(tmpdir))

        return

    def get_dataset_names(self, dataset_paths):
        """
        """
        dataset_names = []
        dataset_paths = [
            dataset_path[2:] if dataset_path.startswith('./') else dataset_path
            for dataset_path in dataset_paths
        ]

        if len(dataset_paths) > 1:
            common_path = os.path.commonpath(dataset_paths)
            dataset_paths = [
                dataset_path[len(common_path) + 1:]
                for dataset_path in dataset_paths
            ]

        for dataset_path in dataset_paths:
            dataset_name = '__'.join(dataset_path.split('/'))
            dataset_name = dataset_name.replace('.', '_')
            dataset_names.append(dataset_name)

        return dataset_names

    def num_classes(self):
        """
        """
        return len(self.out_charset)

    def get_batch(self):
        """
        """
        # Get datasets
        datasets = []
        batch_sizes = []

        for i, (ds_name, ds_path, ds_portion) \
                in enumerate(zip(self.dataset_names,
                                 self.dataset_paths,
                                 self.dataset_portions)):

            # Extract
            if self.concat_batch:
                _batch_size = max(int(self.batch_size * ds_portion), 1) \
                        if i < len(self.dataset_names)-1 \
                        else max(self.batch_size - sum(batch_sizes), 1)
                batch_sizes.append(_batch_size)

            else:
                _batch_size = self.batch_size

            _data_files = glob.glob(ds_path, recursive=True)

            _dataset = tf.data.Dataset.list_files(
                _data_files, shuffle=self.shuffle_and_repeat, seed=self.seed)
            _dataset = _dataset.interleave(self.dataset_class,
                                           cycle_length=self.num_cpus,
                                           num_parallel_calls=self.num_cpus)

            if self.worker_index is not None:
                _dataset = _dataset.shard(self.num_gpus, self.worker_index)

            if self.shuffle_and_repeat:
                _dataset = _dataset.apply(
                    tf.contrib.data.shuffle_and_repeat(buffer_size=_batch_size *
                                                       self.buffer_size,
                                                       seed=self.seed))

            # Trasform
            _dataset = _dataset.map(lambda *e: self.parse_fn(*e, ds_name),
                                    num_parallel_calls=self.num_cpus)

            if self.preprocess_image:
                _dataset = _dataset.map(self.preprocess_fn,
                                        num_parallel_calls=self.num_cpus)

            _dataset = _dataset.filter(self.filter_fn)
            _dataset = _dataset.batch(_batch_size)

            datasets.append(_dataset)

        # Load
        if self.concat_batch:
            batches = []

            for _dataset in datasets:
                _dataset = _dataset.apply(
                    tf.contrib.data.prefetch_to_device(self.input_device, 2))

                _iterator = _dataset.make_initializable_iterator()
                tf.add_to_collection(self.iterator_name, _iterator.initializer)
                _batch = _iterator.get_next()
                batches.append(_batch)

            batch = [tf.concat(elements, axis=0)
                     for elements
                     in zip(*batches)] \
                if len(batches) > 1 \
                else batches[0]

            print('DATASET BATCHES : {} = {}'.format(
                ' + '.join([str(size) for size in batch_sizes]),
                sum(batch_sizes)))

        else:
            concatted = datasets[0]

            for i in range(1, len(datasets)):
                concatted = concatted.concatenate(datasets[i])

            concatted = concatted.apply(
                tf.data.experimental.prefetch_to_device(self.input_device, 2))

            iterator = concatted.make_initializable_iterator()
            tf.add_to_collection(self.iterator_name, iterator.initializer)
            batch = iterator.get_next()

        image, label, length, text, filename, dataset_name = \
            batch
        label = tf.deserialize_many_sparse(label, tf.int64)
        label = tf.cast(label, tf.int32)

        batch = Batch(image, label, length, text, filename, dataset_name)

        return batch

    def tfrecord_parse_fn(self, example, dataset_name):
        """
        """
        feature_map = {
            'image/encoded':
                tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/width':
                tf.FixedLenFeature([1], dtype=tf.int64, default_value=1),
            'image/filename':
                tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'text/string':
                tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'text/length':
                tf.FixedLenFeature([1], dtype=tf.int64, default_value=1)
        }
        features = tf.parse_single_example(example, feature_map)

        image = tf.image.decode_image(features['image/encoded'],
                                      channels=self.img_channels)
        image.set_shape([None, None, self.img_channels])
        length = features['text/length']
        text = features['text/string']

        filename = features['image/filename']
        label = self.indexing_fn(text, length, DELIMITER)
        label = tf.serialize_sparse(label)

        return image, label, length, text, filename, dataset_name

    def txt_parse_fn(self, line, dataset_name):
        """
        """
        line = tf.strings.split([line], sep='\t', maxsplit=1)
        line = tf.sparse.to_dense(line, default_value='')

        filename = tf.reshape(line[0][0], [])
        text = tf.reshape(line[0][1], [])

        length = tf.reshape(tf.py_func(text_length, [text], tf.int64), [-1])

        text = tf.py_func(self.join_with_delimiter, [text], tf.string)
        text = tf.strings.join([text, EOS_TOKEN], separator='')

        image_content = tf.read_file(filename)
        image = tf.image.decode_image(image_content, channels=self.img_channels)
        image.set_shape([None, None, self.img_channels])

        label = self.indexing_fn(text, length, DELIMITER)
        label = tf.serialize_sparse(label)

        return image, label, length, text, filename, dataset_name

    def join_with_delimiter(self, string):
        """
        """
        string = string.decode(encoding='utf-8')
        string = DELIMITER.join([c for c in list(string)])
        string = string + DELIMITER

        return string

    def indexing_fn(self, text, length, sep):
        """
        """
        charlist = tf.strings.split([text], sep=sep)

        if self.exclude_eos:
            length = tf.concat([[1], length], axis=0)
            charlist = tf.sparse.slice(charlist, [0, 0], length)

        label = self.table.lookup(charlist)
        label = tf.sparse.reshape(label, [-1])

        return label

    def preprocess_fn(self, image, *args):
        """
        """
        image = self.preprocess_image(image, self.is_train)

        return (image,) + args

    def filter_fn(self, image, label, length, *args):
        """
        """
        if not self.exclude_eos:
            length = length + 1

        return tf.less_equal(length, self.label_maxlen)[0]

    def _default_preprocess_image(self, image, is_train=True):
        """
        """
        return image
