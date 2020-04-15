"""Construct a classification model_fn that can be used to instantiate
a CS-1 compatible tf.estimator.Estimator object for training.
"""

import tensorflow as tf
from hooks import TrainingHook

def model_fn(features, labels, mode, params):
    """Describe the model to the TensorFlow estimator."""

    if params['hardware'] == 'GPU':
        from build_model import build_model_1to1_gpu as build_model
    elif params['hardware'] == 'CS-1':
        from build_model import build_model_1to1_cs1 as build_model

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_evaluate = (mode == tf.estimator.ModeKeys.EVAL)
    is_predict  = (mode == tf.estimator.ModeKeys.PREDICT)

    if is_training or is_evaluate or is_predict:
        keras_model = build_model(params, inputs=features)
        logits = keras_model.output
        predictions = tf.argmax(logits, -1)

        tf.compat.v1.train.get_or_create_global_step()
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        hook_list = []

        eval_metric_ops = None
        logging_hook = None
        if params['hardware'] == 'GPU':
            accuracy = tf.compat.v1.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                name='accuracy_op',
            )

            eval_metric_ops = dict(accuracy=accuracy)
            tf.compat.v1.summary.scalar('accuracy', accuracy[1])

            logging_hook = tf.estimator.LoggingTensorHook(
                {"loss": loss, "accuracy": accuracy[1]},
                every_n_secs=5,
            )

            hook_list.append(logging_hook)
            if is_training:
                hook_list.append(TrainingHook(params, loss))

        train_op = None
        if is_training:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate'])
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss,
                    global_step=tf.compat.v1.train.get_global_step(),
                )

        estimator = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=hook_list,
        )

        return estimator
