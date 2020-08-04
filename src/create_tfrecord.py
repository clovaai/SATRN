"""
Copyright (c) 2020-present NAVER Corp.
MIT license

Usage:
python create_tfrecord.py
    --input_dir=<input_base_dir> \
    --gt_path=<gt_file_path> \
    --output_dir=<output_dir> \
    --charset_path=<charset_file_path> \
    --dataset_name=<dataset_name> \
    --num_shards=<num_shards> \
    --num_processes=<num_processes>
"""
import os
import re
import io
import glob
import math
import fire
import time
import tensorflow as tf
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from constant import DELIMITER, EOS_TOKEN


def gen_data(gt_exp,
             output_dir,
             dataset_name,
             images_per_shard=10000,
             num_processes=1,
             lowercase=False,
             alphanumeric=False):
    """
    """
    # Parse gt.txt
    img_filenames, texts = parse_gt(gt_exp, lowercase, alphanumeric)

    # Prepare stuff
    num_shards = (len(img_filenames) - 1) // images_per_shard + 1
    num_digits = math.ceil(math.log10(num_shards - 1)) \
        if num_shards != 1 \
        else 1
    shard_format = '%0' + ('%d' % num_digits) + 'd'
    num_of_data = 0
    os.makedirs(output_dir, exist_ok=True)

    # Multiprocessing
    shards_per_process = max(num_shards // num_processes, 1)
    futures = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for pid in range(num_processes):
            start_shard = shards_per_process * pid
            end_shard = shards_per_process * (pid+1) \
                if pid != num_processes-1 \
                else num_shards
            _num_shards = end_shard - start_shard
            start_example = start_shard * images_per_shard
            end_example = end_shard * images_per_shard
            future = executor.submit(
                process_fn,
                pid,
                start_shard,
                _num_shards,
                img_filenames[start_example:end_example],
                texts[start_example:end_example],
                images_per_shard,
                shard_format,
                output_dir,
                dataset_name,
            )
            futures.append(future)

    # Join processes
    num_of_data = sum([future.result() for future in futures])

    return num_of_data


def process_fn(process_id, start_shard, num_shards, image_filenames, texts,
               images_per_shard, shard_format, output_dir, dataset_name):
    """
    """
    print('[+] Process ID : {} START'.format(process_id))
    num_of_data = 0

    for i in range(num_shards):
        start = images_per_shard * i
        end = images_per_shard * (i + 1)
        out_filename = dataset_name+'-words-' + \
            (shard_format % (i+start_shard)) + \
            '.tfrecord'
        out_filename = os.path.join(output_dir, out_filename)

        if os.path.isfile(out_filename):
            continue

        print('PID : ' + str(process_id) + '\t' + str(i) + 'of' +
              str(num_shards) + '[' + str(start) + ':' + str(end) + ']' +
              out_filename)

        num_of_examples_per_shard = gen_shard(
            image_filenames[start:end],
            texts[start:end],
            out_filename,
            dataset_name,
        )
        num_of_data += num_of_examples_per_shard

    print('[+] Process ID : {} END'.format(process_id))

    return num_of_data


def gen_shard(image_filenames,
              texts,
              output_filename,
              dataset_name,
              verbose=True):
    """
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    num_of_examples_per_shard = 0

    for filename, text in zip(image_filenames, texts):
        if os.stat(filename).st_size == 0:
            print('SKIPPING', filename)
            continue

        try:
            image_data, height, width = get_image(filename)

            if verbose and num_of_examples_per_shard < 5:
                print('=' * 70)
                print('IMG PATH :\t{}'.format(filename))
                print('TEXT :\t\t{}'.format(text))

            example = make_example(filename, image_data, text, height, width)
            writer.write(example.SerializeToString())
            num_of_examples_per_shard += 1

        except Exception as e:
            print('ERROR ' + str(e), filename)

    writer.close()

    return num_of_examples_per_shard


def parse_gt(gt_exp, lowercase, alphanumeric, with_spe=False):
    """
    """
    gt_paths = glob.glob(gt_exp, recursive=True)
    filenames, texts = [], []

    for gt_path in gt_paths:
        gt_dir = os.path.dirname(gt_path)

        f = open(gt_path)

        for line in f:
            line = line.strip()

            if line.find('\t') > 0:
                filename, text = line.split('\t')
                filename = filename.strip()
                text = text.strip()

            else:
                filename = line
                text = ''

            if alphanumeric and \
                    not re.match('^[a-zA-Z0-9 ]*$', text):
                continue

            if with_spe and \
                    re.match('^[a-zA-Z0-9 ]*$', text):
                continue

            if lowercase:
                text = text.lower()

            filenames.append(os.path.join(gt_dir, filename))
            texts.append(text)
        f.close()

    return filenames, texts


def get_image(filename):
    """
    """
    image_data = open(filename, 'rb').read()
    image = Image.open(io.BytesIO(image_data))
    width, height = image.size

    return image_data, height, width


def make_example(filename, image_data, text, height, width):
    """
    """
    # Join with delimiter, Add EOS Token
    length = len(text)
    text = DELIMITER.join([c for c in text] + [EOS_TOKEN])
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
            'image/height': _int64_feature([height]),
            'image/width': _int64_feature([width]),
            'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
            'text/string': _bytes_feature(tf.compat.as_bytes(text)),
            'text/length': _int64_feature([length])
        }))

    return example


def _int64_feature(values):
    """
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    """
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def main(argv=None):
    start_time = time.time()
    num_of_data = fire.Fire(gen_data)
    print('[+] NUM OF DATA : {}'.format(num_of_data))
    print('[+] CONVERSION TIME : {} min'.format(
        (time.time() - start_time) / 60))


if __name__ == '__main__':
    main()
