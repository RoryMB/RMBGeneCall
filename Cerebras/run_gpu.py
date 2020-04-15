#!/usr/bin/env python3

"""GPU Estimator."""

import functools
import logging
import os
import tensorflow as tf
from get_arguments import get_arguments
from input_fn import input_fn_1to1 as input_fn
from model_fn import model_fn

def logger(prefix):
    """Initialize TensorFlow logging."""

    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-2s - %(levelname)-2s - %(message)s', '%Y-%m-%d %H:%M:%S')

    # Log to file
    fh = logging.FileHandler(prefix + 'estimator.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
    return logging.getLogger(__name__)

def main(args):
    """Estimator main function."""

    file_prefix = args.inpfx
    if file_prefix:
        file_prefix = file_prefix + '-'

    logger(file_prefix)

    params = {}
    params['file_prefix'] = file_prefix
    params['model_dir'] = args.model_dir
    params['data_dir'] = args.data_dir

    params['input_sizes'] = (3294, 1) # (183, 1, 1, 1, 1, 2)
    params['dropout'] = 0.4
    params['mode'] = args.mode
    params['epochs'] = args.epochs
    params['batch_size'] = args.batch_size
    params['learning_rate'] = args.learning_rate
    params['log_frequency'] = args.log_frequency
    params['shuffle_buffer'] = 1500

    params['hardware'] = 'GPU'
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.gpu

    save_checkpoints_steps = 1
    log_step_count_steps = 1
    train_steps = 1 # metadata['train_samples'] // params['batch_size'] * params['epochs']
    evaluate_steps = 1 # metadata['val_samples'] // params['batch_size']

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=log_step_count_steps,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params['model_dir'],
        config=config,
        params=params
    )

    if params['mode'] == 'train':
        estimator.train(input_fn=input_fn, steps=train_steps)

    if params['mode'] == 'evaluate':
        evaluate_result = estimator.evaluate(input_fn=input_fn, steps=evaluate_steps)
        print('RMBDEBUG global step: %7d'   % evaluate_result['global_step'])
        print('RMBDEBUG accuracy:    %7.2f' % round(evaluate_result['accuracy'] * 100.0, 2))
        print('RMBDEBUG loss:        %7.2f' % round(evaluate_result['loss'], 2))

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    arguments = get_arguments()
    main(arguments)
