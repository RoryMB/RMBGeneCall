import glob
import json
import logging
import os
import sys
import time

# ours
from get_arguments import get_arguments
from tfrecord_data import input_fn
from hybrid_model  import model_fn

#import numpy as np
import tensorflow as tf

#                               # parameterize these ??????????????????
DROPOUT = 0.20                  # ?????????????????????????????????????
SHUFFLE_BUFFER = 1500           # ?????????????????????????????????????
NBR_CLASSES = 2                 # ?????????????????????????????????????

def logger(prefix):
    """ """
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-2s - %(levelname)-2s - %(message)s', "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(prefix + 'attn_bin_estimator.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    return logging.getLogger(__name__)


def qualify_path(directory):
    """Generate fully qualified path name from input file name."""
    return os.path.abspath(directory)


def main(args):
    """ """
    file_prefix = args.inpfx
    if file_prefix:
        file_prefix = file_prefix + '-'
    logger(file_prefix)

    params = {}
    params['model_dir'] = qualify_path(args.model_dir)
    params['data_dir'] = qualify_path(args.data_dir)
    params['file_prefix'] = file_prefix
    params['epochs'] = args.epochs
    params['batch_size'] = args.batch_size
    params['learning_rate'] = args.learning_rate
    params['log_frequency'] = args.log_frequency
    params['xla'] = args.xla

    params['shuffle_buffer'] = SHUFFLE_BUFFER
    params['nbr_classes'] = NBR_CLASSES
    params['dropout'] = DROPOUT
    params['input_sizes'] = (6212, 1)

    eval = args.eval
    train = args.eval
    epochs = args.epochs
    data_dir = params['data_dir']
    model_dir = params['model_dir']
    batch_size = args.batch_size

    print("*" * 130)
    print(f"Batch size is {batch_size}")
    print(f"Number of epochs: {epochs}")
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print("params:", params)
    print("args:", args)
    print("*" * 130)

    # build estimator
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params)

    # train model
    if train:
        print("\nTraining...")
        _input_fn = lambda: input_fn(data_dir, batch_size, is_training=True, params=params)
        model.train(input_fn=_input_fn)                     # train_steps defaults to None
        #model.train(input_fn=_input_fn, steps=train_steps) # ????????????? rely on epochs
        print("Training complete")

    # evaluate model
    if eval:
        print("\nEvaluating...")
        _eval_input_fn = lambda: input_fn(data_dir, batch_size, is_training=False, params=params)
        eval_result = model.evaluate(input_fn=_eval_input_fn)

        print("global step:%7d" % eval_result['global_step'])
        print("accuracy:   %7.2f" % round(eval_result['accuracy'] * 100.0, 2))
        print("loss:       %7.2f" % round(eval_result['loss'], 2))
        print("Evaluation complete")

##______________________________________________________________________________
if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments)
