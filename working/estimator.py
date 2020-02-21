#!/usr/bin/env python3

"""CS-1 / GPU compatible TensorFlow Estimator."""

import functools
import logging
import os
import tensorflow as tf
from get_arguments import get_arguments
from input_fn import input_fn
from model_fn import model_fn

# Cerebras
try:
    from cerebras.tf.cs_estimator import CerebrasEstimator as CommonEstimator
    from cerebras.tf.run_config import CSRunConfig as CommonRunConfig
    from cerebras.tf.cs_slurm_cluster_resolver import CSSlurmClusterResolver
    CEREBRAS_ENV = True
except:
    print("Cerebras support is not available")
    CEREBRAS_ENV = False
    class CommonEstimator(tf.estimator.Estimator):
        def __init__(self, use_cs=None, **kwargs):
            super(CommonEstimator, self).__init__(**kwargs)
    class CommonRunConfig(tf.estimator.RunConfig):
        def __init__(self, cs_ip=None, **kwargs):
            super(CommonRunConfig, self).__init__(**kwargs)

def validate_arguments(mode_list, is_cerebras, params_dict):
    """Estimator script argument/environment validation """
    if 'validate_only' in mode_list or 'compile_only' in mode_list:
        if not is_cerebras:
            tf.compat.v1.logging.error("validate_only and compile_only not available")
            return False

    if is_cerebras and 'train' in mode_list:
        if not params['cs_ip']:
            tf.compat.v1.logging.error("--cs_ip is required when training on the CS-1")
            return False

        if ':' not in params['cs_ip']:
            params['cs_ip'] += ':9000'              # why?

    return True

def logger(prefix):
    """Initialize TensorFlow logging."""

    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-2s - %(levelname)-2s - %(message)s', "%Y-%m-%d %H:%M:%S")

    # Log to file
    fh = logging.FileHandler(prefix + 'estimator.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    # Log to terminal
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    return logging.getLogger(__name__)

def main(args):
    """Estimator main function."""

    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.gpu

    file_prefix = args.inpfx
    if file_prefix:
        file_prefix = file_prefix + '-'
    logger(file_prefix)

    params = {}
    params['epochs'] = args.epochs
    params['data_dir'] = args.data_dir
    params['model_dir'] = args.model_dir
    params['batch_size'] = args.batch_size
    params['file_prefix'] = file_prefix
    params['mode'] = args.mode
    params['learning_rate'] = args.learning_rate
    params['log_frequency'] = args.log_frequency
    params['shuffle_buffer'] = 1500
    params['dropout'] = 0.4
    params['input_sizes'] = (3294, 1)
    params['cerebras'] = CEREBRAS_ENV
    params['cs_ip'] = args.cs_ip

    epochs = params['epochs']
    data_dir = params['data_dir']
    model_dir = params['model_dir']
    batch_size = params['batch_size']

    print("*" * 130)
    print(f"Batch size is {batch_size}")
    print(f"Number of epochs: {epochs}")
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print("params:", params)
    print("args:", args)
    print("*" * 130)

    if not validate_arguments(args.mode, CEREBRAS_ENV, params):
        print("Unable to continue, correct arguments or environment")
        return

    # Build estimator
    config = CommonRunConfig(
        cs_ip=params['cs_ip'],
        # save_checkpoints_steps=10,
        # log_step_count_steps=10
    )

    model = CommonEstimator(
        use_cs=params['cerebras'],
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        params=params
    )

    # Predict
    if 'predict' in args.mode:
        errstr = 'PREDICT mode not yet implemented'
        tf.compat.v1.logging.error(errstr)
        raise NotImplementedError(errstr)

    # Train
    if 'train' in args.mode:
        if CEREBRAS_ENV:
            PORT_BASE = 23111
            slurm_cluster_resolver = CSSlurmClusterResolver(port_base=PORT_BASE)
            cluster_spec = slurm_cluster_resolver.cluster_spec()
            task_type, task_id = slurm_cluster_resolver.get_task_info()
            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': cluster_spec.as_dict(),
                'task': {
                    'type': task_type,
                    'index': task_id
                }
            })

            os.environ['SEND_BLOCK'] = '16384'      # what do these stmts do
            os.environ['RECV_BLOCK'] = '16384'

        print("\nTraining...")
        _input_fn = lambda: input_fn(data_dir, batch_size, is_training=True, params=params)
        model.train(input_fn=_input_fn)
        print("Training complete")

    # Evaluate
    if 'eval' in args.mode:
        print("\nEvaluating...")
        _eval_input_fn = lambda: input_fn(data_dir, batch_size, is_training=False, params=params)
        eval_result = model.evaluate(input_fn=_eval_input_fn)

        print("global step:%7d" % eval_result['global_step'])
        print("accuracy:   %7.2f" % round(eval_result['accuracy'] * 100.0, 2))
        print("loss:       %7.2f" % round(eval_result['loss'], 2))
        print("Evaluation complete")

    if 'compile_only' in args.mode or 'validate_only' in args.mode:
        print("CS-1 preprocessing...")
        validate_only = 'validate_only' in args.mode
        # est_input_fn = lambda: input_fn(data_dir, batch_size, is_training=False, params=params)
        est_input_fn = functools.partial(input_fn, data_dir, batch_size, is_training=False, params=params)
        model.compile(est_input_fn, validate_only=validate_only)
        print("CS-1 preprocessing complete")

if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments)
