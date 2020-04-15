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
    import json
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

    params['input_sizes'] = (183, 1, 1, 1, 1, 2)
    params['dropout'] = 0.4
    params['mode'] = args.mode
    params['epochs'] = args.epochs
    params['batch_size'] = args.batch_size
    params['learning_rate'] = args.learning_rate
    params['log_frequency'] = args.log_frequency
    params['shuffle_buffer'] = 1500

    params['cerebras'] = CEREBRAS_ENV
    params['cs_ip'] = args.cs_ip
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.gpu

    save_checkpoints_steps = 1
    log_step_count_steps = 1
    train_steps = 1 # metadata['train_samples'] // params['batch_size'] * params['epochs']
    evaluate_steps = 1 # metadata['val_samples'] // params['batch_size']

    if params['mode'] == 'compile_only' or params['mode'] == 'validate_only':
        if not CEREBRAS_ENV:
            tf.compat.v1.logging.error("validate_only and compile_only only available on CS-1")
            exit()

    if CEREBRAS_ENV and params['mode'] == 'train':
        if not params['cs_ip']:
            tf.compat.v1.logging.error("--cs_ip is required when training on CS-1")
            exit()

        if ':' not in params['cs_ip']:
            params['cs_ip'] += ':9000'

    config = CommonRunConfig(
        cs_ip=params['cs_ip'],
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=log_step_count_steps,
    )

    estimator = CommonEstimator(
        use_cs=params['cerebras'],
        model_fn=model_fn,
        model_dir=params['model_dir'],
        config=config,
        params=params
    )

    if params['mode'] == 'train':
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

            os.environ['SEND_BLOCK'] = '16384'
            os.environ['RECV_BLOCK'] = '16384'

        estimator.train(input_fn=input_fn, steps=train_steps)

    if params['mode'] == 'evaluate':
        evaluate_result = estimator.evaluate(input_fn=input_fn, steps=evaluate_steps)
        print('RMBDEBUG global step: %7d'   % evaluate_result['global_step'])
        print('RMBDEBUG accuracy:    %7.2f' % round(evaluate_result['accuracy'] * 100.0, 2))
        print('RMBDEBUG loss:        %7.2f' % round(evaluate_result['loss'], 2))

    if params['mode'] == 'compile_only' or params['mode'] == 'validate_only':
        estimator.compile(input_fn=input_fn, validate_only=(params['mode']=='validate_only'))

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    arguments = get_arguments()
    main(arguments)
