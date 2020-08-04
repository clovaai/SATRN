"""
Copyright (c) 2020-present NAVER Corp.
MIT license

Usage:
python train.py --config_file=<config_file_path>
"""

import os
import time
import random
import logging
from logging import handlers, StreamHandler
from psutil import virtual_memory

import fire
import numpy as np
import tensorflow as tf

from flags import Flags
from constant import DELIMITER
from dataset import DatasetLodaer
from utils import \
        load_charset, get_optimizer, \
        get_network, get_session_config, get_string, \
        adjust_string, count_available_gpus, \
        single_tower, validate, get_session, get_scaffold, get_init_trained

tf.logging.set_verbosity(tf.logging.ERROR)


def _average_gradients(tower_grads):
    """ Average gradients from multiple towers.
    """
    average_grads = []

    for grads_and_vars in zip(*tower_grads):
        grads = tf.stack([g for g, _ in grads_and_vars])
        grad = tf.reduce_mean(grads, 0)
        v = grads_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def _clip_gradients(grads, grad_clip_norm):
    """ Clip gradients with global norm.
    """
    g, v = zip(*grads)
    g, global_norm = tf.clip_by_global_norm(g, grad_clip_norm)
    clipped_grads = list(zip(g, v))

    return clipped_grads, global_norm


def create_model_dir(model_dir):
    """ Create model directory
    """
    os.makedirs(os.path.join(model_dir, 'best_models'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'valid'), exist_ok=True)

    return model_dir


def get_logger(model_dir, name):
    """ Get stdout, file logger
    """
    logger = logging.getLogger(name)
    logger.handlers = []

    # Stdout
    streamHandler = StreamHandler()
    logger.addHandler(streamHandler)

    # File
    log_dir = os.path.join(model_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, '{}.log'.format(name))
    fileHandler = handlers.RotatingFileHandler(log_path,
                                               maxBytes=1024 * 1024,
                                               backupCount=10)
    logger.addHandler(fileHandler)

    logger.setLevel(logging.INFO)

    return logger


def set_seed(seed):
    """ Set random seed
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        print('[+] Random seed is set to {}'.format(seed))

    return


def log_formatted(logger, *logs):
    """ Print multiple lines log.
    """
    logger.info('=' * 70)
    logger.info(' ' * 70)

    for log in logs:
        logger.info(log)

    logger.info(' ' * 70)
    logger.info('=' * 70)

    return


def main(config_file):
    """ Train text recognition network
    """
    # Parse configs
    FLAGS = Flags(config_file).get()

    # Set directory, seed, logger
    model_dir = create_model_dir(FLAGS.model_dir)
    logger = get_logger(model_dir, 'train')
    best_model_dir = os.path.join(model_dir, 'best_models')
    set_seed(FLAGS.seed)

    # Print configs
    flag_strs = [
        '{}:\t{}'.format(name, value)
        for name, value in FLAGS._asdict().items()
    ]
    log_formatted(logger, '[+] Model configurations', *flag_strs)

    # Print system environments
    num_gpus = count_available_gpus()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024**3)
    log_formatted(logger, '[+] System environments',
                  'The number of gpus : {}'.format(num_gpus),
                  'The number of cpus : {}'.format(num_cpus),
                  'Memory Size : {}G'.format(mem_size))

    # Get optimizer and network
    global_step = tf.train.get_or_create_global_step()
    optimizer, learning_rate = get_optimizer(FLAGS.train.optimizer, global_step)
    out_charset = load_charset(FLAGS.charset)
    net = get_network(FLAGS, out_charset)
    is_ctc = (net.loss_fn == 'ctc_loss')

    # Multi tower for multi-gpu training
    tower_grads = []
    tower_extra_update_ops = []
    tower_preds = []
    tower_gts = []
    tower_losses = []
    batch_size = FLAGS.train.batch_size
    tower_batch_size = batch_size // num_gpus

    val_tower_outputs = []
    eval_tower_outputs = []

    for gpu_indx in range(num_gpus):

        # Train tower
        print('[+] Build Train tower GPU:%d' % gpu_indx)
        input_device = '/gpu:%d' % gpu_indx

        tower_batch_size = tower_batch_size \
            if gpu_indx < num_gpus-1 \
            else batch_size - tower_batch_size * (num_gpus-1)

        train_loader = DatasetLodaer(
            dataset_paths=FLAGS.train.dataset_paths,
            dataset_portions=FLAGS.train.dataset_portions,
            batch_size=tower_batch_size,
            label_maxlen=FLAGS.label_maxlen,
            out_charset=out_charset,
            preprocess_image=net.preprocess_image,
            is_train=True,
            is_ctc=is_ctc,
            shuffle_and_repeat=True,
            concat_batch=True,
            input_device=input_device,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            worker_index=gpu_indx,
            use_rgb=FLAGS.use_rgb,
            seed=FLAGS.seed,
            name='train')

        tower_output = single_tower(net,
                                    gpu_indx,
                                    train_loader,
                                    out_charset,
                                    optimizer,
                                    name='train',
                                    is_train=True)
        tower_grads.append([x for x in tower_output.grads if x[0] is not None])
        tower_extra_update_ops.append(tower_output.extra_update_ops)
        tower_preds.append(tower_output.prediction)
        tower_gts.append(tower_output.text)
        tower_losses.append(tower_output.loss)

        # Print network structure
        if gpu_indx == 0:
            param_stats = tf.profiler.profile(tf.get_default_graph())
            logger.info('total_params: %d\n' % param_stats.total_parameters)

        # Valid tower
        print('[+] Build Valid tower GPU:%d' % gpu_indx)
        valid_loader = DatasetLodaer(dataset_paths=FLAGS.valid.dataset_paths,
                                     dataset_portions=None,
                                     batch_size=FLAGS.valid.batch_size //
                                     num_gpus,
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
                                     name='valid')

        val_tower_output = single_tower(net,
                                        gpu_indx,
                                        valid_loader,
                                        out_charset,
                                        optimizer=None,
                                        name='valid',
                                        is_train=False)

        val_tower_outputs.append(
            (val_tower_output.loss, val_tower_output.prediction,
             val_tower_output.text, val_tower_output.filename,
             val_tower_output.dataset))

    # Aggregate gradients
    losses = tf.reduce_mean(tower_losses)
    grads = _average_gradients(tower_grads)

    with tf.control_dependencies(tower_extra_update_ops[-1]):
        if FLAGS.train.optimizer.grad_clip_norm is not None:
            grads, global_norm = _clip_gradients(
                grads, FLAGS.train.optimizer.grad_clip_norm)
            tf.summary.scalar('global_norm', global_norm)

        train_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Define config, scaffold
    saver = tf.train.Saver()
    sess_config = get_session_config()
    scaffold = get_scaffold(saver, FLAGS.train.tune_from, 'train')
    restore_model = get_init_trained()

    # Define validation saver, summary writer
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    val_summary_op = tf.summary.merge(
        [s for s in summaries if 'valid' in s.name])
    val_summary_writer = {
        dataset_name:
        tf.summary.FileWriter(os.path.join(model_dir, 'valid', dataset_name))
        for dataset_name in valid_loader.dataset_names
    }
    val_summary_writer['total_valid'] = tf.summary.FileWriter(
        os.path.join(model_dir, 'valid', 'total_valid'))
    val_saver = tf.train.Saver(max_to_keep=len(valid_loader.dataset_names) + 1)
    best_val_err_rates = {}
    best_steps = {}

    # Training
    print('[+] Make Session...')

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=model_dir,
            scaffold=scaffold,
            config=sess_config,
            save_checkpoint_steps=FLAGS.train.save_steps,
            save_checkpoint_secs=None,
            save_summaries_steps=FLAGS.train.summary_steps,
            save_summaries_secs=None,
    ) as sess:

        log_formatted(logger, 'Training started!')
        _step = 0
        train_t = 0
        start_t = time.time()

        while _step < FLAGS.train.max_num_steps \
                and not sess.should_stop():

            # Train step
            step_t = time.time()
            [step_loss, _, _step, preds, gts, lr] = sess.run([
                losses, train_op, global_step, tower_preds[0], tower_gts[0],
                learning_rate
            ])
            train_t += time.time() - step_t

            # Summary
            if _step % FLAGS.valid.steps == 0:

                # Train summary
                train_err = 0.

                for i, (p, g) in enumerate(zip(preds, gts)):
                    s = get_string(p, out_charset, is_ctc=is_ctc)
                    g = g.decode('utf8').replace(DELIMITER, '')

                    s = adjust_string(s, FLAGS.train.lowercase,
                                      FLAGS.train.alphanumeric)
                    g = adjust_string(g, FLAGS.train.lowercase,
                                      FLAGS.train.alphanumeric)
                    e = int(s != g)

                    train_err += e

                    if FLAGS.train.verbose and i < 5:
                        print('TRAIN :\t{}\t{}\t{}'.format(s, g, not bool(e)))

                train_err_rate = \
                    train_err / len(gts)

                # Valid summary
                val_cnts, val_errs, val_err_rates, _ = \
                    validate(sess,
                             _step,
                             val_tower_outputs,
                             out_charset,
                             is_ctc,
                             val_summary_op,
                             val_summary_writer,
                             val_saver,
                             best_val_err_rates,
                             best_steps,
                             best_model_dir,
                             FLAGS.valid.lowercase,
                             FLAGS.valid.alphanumeric)

                # Logging
                log_strings = ['', '-' * 28 + ' VALID_DETAIL ' + '-' * 28, '']

                for dataset in sorted(val_err_rates.keys()):
                    if dataset == 'total_valid':
                        continue

                    cnt = val_cnts[dataset]
                    err = val_errs[dataset]
                    err_rate = val_err_rates[dataset]
                    best_step = best_steps[dataset]

                    s = '%s : %.2f%%(%d/%d)\tBEST_STEP : %d' % \
                        (dataset, (1.-err_rate)*100, cnt-err, cnt, best_step)

                    log_strings.append(s)

                elapsed_t = float(time.time() - start_t) / 60
                remain_t = (elapsed_t / (_step+1)) * \
                    (FLAGS.train.max_num_steps - _step - 1)
                log_formatted(
                    logger, 'STEP : %d\tTRAIN_LOSS : %f' % (_step, step_loss),
                    'ELAPSED : %.2f min\tREMAIN : %.2f min\t'
                    'STEP_TIME: %.1f sec' %
                    (elapsed_t, remain_t, float(train_t) / (_step + 1)),
                    'TRAIN_SEQ_ERR : %f\tVALID_SEQ_ERR : %f' %
                    (train_err_rate, val_err_rates['total_valid']),
                    'BEST_STEP : %d\tBEST_VALID_SEQ_ERR : %f' %
                    (best_steps['total_valid'],
                     best_val_err_rates['total_valid']), *log_strings)

        log_formatted(logger, 'Training is completed!')


if __name__ == '__main__':
    fire.Fire(main)
