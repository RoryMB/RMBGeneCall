"""TensorFlow Estimator-compatible input_fn."""

import os
import tensorflow as tf

def input_fn(params):
    """Return dataset iterator."""

    epochs = params['epochs']
    file_prefix = params['file_prefix']
    shuffle_buffer = params['shuffle_buffer']
    data_sz = params['input_sizes']

    file_pattern = os.path.join(
        params['data_dir'],
        params['file_prefix'] + params['mode'] + '*.tfrecords',
    )
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    def _parse_record_fn(raw_record):
        """Decode raw TFRecord into feature and label components."""

        feature_map = {
            'sequence': tf.io.FixedLenFeature([data_sz[0]], dtype=tf.float32),
            'geneLength': tf.io.FixedLenFeature([data_sz[1]], dtype=tf.float32),
            'orfLength': tf.io.FixedLenFeature([data_sz[2]], dtype=tf.float32),
            'genomeGC': tf.io.FixedLenFeature([data_sz[3]], dtype=tf.float32),
            'contigGC': tf.io.FixedLenFeature([data_sz[4]], dtype=tf.float32),
            'labels': tf.io.FixedLenFeature([data_sz[5]], dtype=tf.int64),
        }

        record_features = tf.io.parse_single_example(raw_record, feature_map)

        sequence = record_features['sequence']
        geneLength = record_features['geneLength']
        orfLength = record_features['orfLength']
        genomeGC = record_features['genomeGC']
        contigGC = record_features['contigGC']

        data = {
            'sequence':sequence,
            'geneLength':geneLength,
            'orfLength':orfLength,
            'genomeGC':genomeGC,
            'contigGC':contigGC,
        }
        labels = tf.cast(record_features['labels'], dtype=tf.int32)

        return data, labels

    dataset = dataset.prefetch(buffer_size=batch_size)

    if params['mode'] == 'train':
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.repeat(count=None)

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda raw_record: _parse_record_fn(raw_record),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=True,
        )
    )

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset
