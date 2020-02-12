"""Construct a classification model_fn that can be used to instantiate
a CS-1 compatible tf.estimator.Estimator object for training.
"""

import estimator_hooks as hooks
import tensorflow as tf
from build_model import build_model

def model_fn(features, labels, mode, params):
    """Describe the model to the TensorFlow estimator."""
    XLA = params['xla']
    LR = params['learning_rate']

    # From Cerebras MNIST hybrid_model.py
    tf.compat.v1.set_random_seed(0)
    loss = None
    train_op = None
    logging_hook = None
    training_hook = None
    eval_metric_ops = None

    # Living in the past?
    # get_global_step_fn = tf.compat.v1.train.get_global_step
    get_collection_fn = tf.compat.v1.get_collection
    set_verbosity_fn = tf.compat.v1.logging.set_verbosity
    # optimizer_fn = tf.compat.v1.train.MomentumOptimizer
    optimizer_fn = tf.compat.v1.train.AdamOptimizer
    accuracy_fn = tf.compat.v1.metrics.accuracy
    # loss_fn = tf.compat.v1.losses.sparse_softmax_cross_entropy
    loss_fn = tf.compat.v1.keras.losses.binary_crossentropy

    logging_INFO = tf.compat.v1.logging.INFO
    GraphKeys = tf.compat.v1.GraphKeys
    summary_scalar = tf.compat.v1.summary.scalar

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_evaluate = (mode == tf.estimator.ModeKeys.EVAL)
    is_predict = (mode == tf.estimator.ModeKeys.PREDICT)

    # No obvious transformations needed
    inputs = features
    keras_model = build_model(params, tensor=inputs)
    logits = keras_model.output
    # Why axis = 1
    predictions = tf.argmax(logits, 1)

    # Label and class weights are concatenated, decouple
    # weights = labels[:,1]
    float_labels = labels[:,0]
    labels = tf.cast(float_labels, dtype=tf.int64)

    if is_training or is_evaluate:
        global_step = tf.train.get_or_create_global_step()
        # loss = loss_fn(labels=labels, logits=logits)
        loss = loss_fn(y_true=labels, y_pred=tf.squeeze(logits))
        hook_list = []

        if not XLA:
            accuracy = accuracy_fn(
                labels=labels,
                predictions=predictions,
                name='accuracy_op')

            eval_metric_ops = dict(accuracy=accuracy)

            summary_scalar('accuracy', accuracy[1])

            set_verbosity_fn(logging_INFO)
            logging_hook = tf.estimator.LoggingTensorHook(
                {"loss": loss, "accuracy": accuracy[1]},
                every_n_iter = 1000) # every_n_secs = 60)

            hook_list.append(logging_hook)

        if is_training:
            optimizer = optimizer_fn(learning_rate=LR)
            update_ops = get_collection_fn(GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss,
                    global_step=tf.compat.v1.train.get_global_step())

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
