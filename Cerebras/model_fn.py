"""Construct a classification model_fn that can be used to instantiate
a CS-1 compatible tf.estimator.Estimator object for training.
"""

import estimator_hooks as hooks
import tensorflow as tf
from build_model import build_model

def model_fn(features, labels, mode, params):
    """Describe the model to the TensorFlow estimator."""
    cerebras = params['cerebras']
    LR = params['learning_rate']

    # From Cerebras MNIST hybrid_model.py
    tf.compat.v1.set_random_seed(0)
    loss = None
    train_op = None
    logging_hook = None
    training_hook = None
    eval_metric_ops = None

    # Living in the past?
    get_global_step_fn = tf.compat.v1.train.get_global_step
    get_collection_fn = tf.compat.v1.get_collection
    set_verbosity_fn = tf.compat.v1.logging.set_verbosity
    optimizer_fn = tf.compat.v1.train.AdamOptimizer
    accuracy_fn = tf.compat.v1.metrics.accuracy
    loss_fn = tf.compat.v1.keras.losses.sparse_categorical_crossentropy

    logging_INFO = tf.compat.v1.logging.INFO
    GraphKeys = tf.compat.v1.GraphKeys
    summary_scalar = tf.compat.v1.summary.scalar

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_evaluate = (mode == tf.estimator.ModeKeys.EVAL)
    is_predict = (mode == tf.estimator.ModeKeys.PREDICT)

    keras_model = build_model(params, inputs=features)
    outputs = keras_model.output
    predictions = tf.argmax(outputs, -1)

    # Label and class weights are concatenated, decouple
    labels = tf.cast(labels, dtype=tf.int64)

    if is_training or is_evaluate:
        global_step = tf.compat.v1.train.get_or_create_global_step()
        print('RMBDEBUG', '*'*50)
        print('features:                  ', features)
        print('labels:                    ', labels)
        print('tf.squeeze(labels).shape:  ', tf.squeeze(labels).shape)
        print('outputs:                   ', outputs)
        print('tf.squeeze(outputs).shape: ', tf.squeeze(outputs).shape)
        print('RMBDEBUG', '*'*50)
        loss = tf.compat.v1.keras.backend.sum(loss_fn(y_true=labels, y_pred=outputs))
        confusion_matrix = tf.math.confusion_matrix(labels[:,0], predictions[:,0])
        hook_list = []

        if not cerebras:
            accuracy = accuracy_fn(
                labels=labels,
                predictions=predictions,
                name='accuracy_op')

            eval_metric_ops = dict(accuracy=accuracy)
            summary_scalar('accuracy', accuracy[1])

            set_verbosity_fn(logging_INFO)
            logging_hook = tf.estimator.LoggingTensorHook(
                {"loss": loss, "accuracy": accuracy[1], "\nconfmat": confusion_matrix},
                # every_n_iter=1000,
                every_n_secs=5,
            )

            hook_list.append(logging_hook)

        if is_training:
            optimizer = optimizer_fn(learning_rate=LR)
            update_ops = get_collection_fn(GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss,
                    global_step=get_global_step_fn())

            training_hook = hooks.TrainingHook(params, loss)
            hook_list.append(training_hook)

        estimator = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=hook_list)

        return estimator
