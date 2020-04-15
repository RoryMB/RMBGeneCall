"""TensorFlow Estimator-compatible input_fn."""

import os
import tensorflow as tf

def input_fn_1to1(params):
    """Return dataset iterator."""

    def _parse_record_fn(raw_record):
        """Decode raw TFRecord into feature and label components."""

        feature_map = {
            'data' : tf.io.FixedLenFeature([params['input_sizes'][0]], dtype=tf.float32),
            'label': tf.io.FixedLenFeature([params['input_sizes'][1]], dtype=tf.int64),
        }

        record_features = tf.io.parse_single_example(raw_record, feature_map)

        data = {
            'sequence': record_features['data']
        }
        labels = tf.cast(record_features['label'], dtype=tf.int32)

        return data, labels

    file_pattern = os.path.join(
        params['data_dir'],
        params['file_prefix'] + params['mode'] + '*.tfrecords',
    )
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.prefetch(buffer_size=params['batch_size'])

    if params['mode'] == 'train':
        dataset = dataset.shuffle(buffer_size=params['shuffle_buffer'])
        dataset = dataset.repeat(count=None)

    dataset = dataset.map(_parse_record_fn, num_parallel_calls=tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=params['batch_size'], drop_remainder=True)

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset

def input_fn_5to2(params):
    """Return dataset iterator."""

    def _parse_record_fn(raw_record):
        """Decode raw TFRecord into feature and label components."""

        feature_map = {
            'sequence'  : tf.io.FixedLenFeature([params['input_sizes'][0]], dtype=tf.float32),
            'geneLength': tf.io.FixedLenFeature([params['input_sizes'][1]], dtype=tf.float32),
            'orfLength' : tf.io.FixedLenFeature([params['input_sizes'][2]], dtype=tf.float32),
            'genomeGC'  : tf.io.FixedLenFeature([params['input_sizes'][3]], dtype=tf.float32),
            'contigGC'  : tf.io.FixedLenFeature([params['input_sizes'][4]], dtype=tf.float32),
            'labels'    : tf.io.FixedLenFeature([params['input_sizes'][5]], dtype=tf.int64),
        }

        record_features = tf.io.parse_single_example(raw_record, feature_map)

        data = {
            'sequence'  : record_features['sequence'],
            'geneLength': record_features['geneLength'],
            'orfLength' : record_features['orfLength'],
            'genomeGC'  : record_features['genomeGC'],
            'contigGC'  : record_features['contigGC'],
        }
        labels = tf.cast(record_features['labels'], dtype=tf.int32)

        return data, labels

    file_pattern = os.path.join(
        params['data_dir'],
        params['file_prefix'] + params['mode'] + '*.tfrecords',
    )
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.prefetch(buffer_size=params['batch_size'])

    if params['mode'] == 'train':
        dataset = dataset.shuffle(buffer_size=params['shuffle_buffer'])
        dataset = dataset.repeat(count=None)

    dataset = dataset.map(_parse_record_fn, num_parallel_calls=tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=params['batch_size'], drop_remainder=True)

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset
