"""
Copyright (c) 2020-present NAVER Corp.
MIT license

Usage:
python eval.py --config_file=<config_file_path>
"""
import os
import fire
import time

import tensorflow as tf
from psutil import virtual_memory

from flags import Flags
from dataset import DatasetLodaer
from utils import \
        count_available_gpus, load_charset, get_network, \
        single_tower, validate, get_session_config, \
        get_init_trained, get_scaffold

tf.logging.set_verbosity(tf.logging.ERROR)


def main(config_file=None):
    """ Run evaluation.
    """
    # Parse Config
    print('[+] Model configurations')
    FLAGS = Flags(config_file).get()
    for name, value in FLAGS._asdict().items():
        print('{}:\t{}'.format(name, value))
    print('\n')

    # System environments
    num_gpus = count_available_gpus()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024**3)
    out_charset = load_charset(FLAGS.charset)
    print('[+] System environments')
    print('The number of gpus : {}'.format(num_gpus))
    print('The number of cpus : {}'.format(num_cpus))
    print('Memory Size : {}G'.format(mem_size))
    print('The number of characters : {}\n'.format(len(out_charset)))

    # Make results dir
    res_dir = os.path.join(FLAGS.eval.model_path)
    os.makedirs(res_dir, exist_ok=True)

    # Get network
    net = get_network(FLAGS, out_charset)
    is_ctc = (net.loss_fn == 'ctc_loss')

    # Define Graph
    eval_tower_outputs = []
    global_step = tf.train.get_or_create_global_step()

    for gpu_indx in range(num_gpus):

        # Get eval dataset
        input_device = '/gpu:%d' % gpu_indx
        print('[+] Build Eval tower GPU:%d' % gpu_indx)

        eval_loader = DatasetLodaer(dataset_paths=FLAGS.eval.dataset_paths,
                                    dataset_portions=None,
                                    batch_size=FLAGS.eval.batch_size
                                    label_maxlen=FLAGS.label_maxlen,
                                    out_charset=out_charset,
                                    preprocess_image=net.preprocess_image,
                                    is_train=False,
                                    is_ctc=is_ctc,
                                    shuffle_and_repeat=False,
                                    concat_batch=False,
                                    input_device=input_device,
                                    num_cpus=num_cpus,
                                    num_gpus=num_gpus,
                                    worker_index=gpu_indx,
                                    use_rgb=FLAGS.use_rgb,
                                    seed=FLAGS.seed,
                                    name='eval')

        eval_tower_output = single_tower(net,
                                         gpu_indx,
                                         eval_loader,
                                         out_charset,
                                         optimizer=None,
                                         name='eval',
                                         is_train=False)

        eval_tower_outputs.append(
            (eval_tower_output.loss, eval_tower_output.prediction,
             eval_tower_output.text, eval_tower_output.filename,
             eval_tower_output.dataset))

    # Summary
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge([s for s in summaries])
    summary_writer = {
        dataset_name: tf.summary.FileWriter(os.path.join(res_dir, dataset_name))
        for dataset_name in eval_loader.dataset_names
    }
    summary_writer['total_valid'] = tf.summary.FileWriter(
        os.path.join(res_dir, 'total_eval'))

    # Define config, scaffold, hooks
    saver = tf.train.Saver()
    sess_config = get_session_config()
    restore_model = get_init_trained()
    scaffold = get_scaffold(saver, None, 'eval')

    # Testing
    with tf.train.MonitoredTrainingSession(scaffold=scaffold,
                                           config=sess_config) as sess:

        # Restore and init.
        restore_model(sess, FLAGS.eval.model_path)
        _step = sess.run(global_step)
        infet_t = 0

        # Run test
        start_t = time.time()
        eval_cnts, eval_errs, eval_err_rates, eval_preds = \
            validate(sess,
                     _step,
                     eval_tower_outputs,
                     out_charset,
                     is_ctc,
                     summary_op,
                     summary_writer,
                     lowercase=FLAGS.eval.lowercase,
                     alphanumeric=FLAGS.eval.alphanumeric)
        infer_t = time.time() - start_t

    # Log
    total_total = 0

    for dataset, result in eval_preds.items():
        res_file = open(os.path.join(res_dir, '{}.txt'.format(dataset)), 'w')
        total = eval_cnts[dataset]
        correct = total - eval_errs[dataset]
        acc = 1. - eval_err_rates[dataset]
        total_total += total

        for f, s, g in result:
            f = f.decode('utf8')

            if FLAGS.eval.verbose:
                print('FILE : ' + f)
                print('PRED : ' + s)
                print('ANSW : ' + g)
                print('=' * 50)

            res_file.write('{}\t{}\n'.format(f, s))

        res_s = 'DATASET : %s\tCORRECT : %d\tTOTAL : %d\tACC : %f' % \
                (dataset, correct, total, acc)
        print(res_s)
        res_file.write(res_s)
        res_file.close()

    eval_loader.flush_tmpfile()
    print('INFER TIME(PER IMAGE) : %f s' % (float(infer_t) / total_total))


if __name__ == '__main__':
    fire.Fire(main)
