"""TensorFlow Estimator-compatible input_fn."""

import os
import tensorflow as tf

def input_fn(data_dir, batch_size, is_training=None, params=None):
    """Return dataset iterator."""
    epochs = params['epochs']
    file_prefix = params['file_prefix']
    shuffle_buffer = params['shuffle_buffer']
    data_sz = params['input_sizes']

    if is_training:
        partition = 'train'
    else:
        partition = 'test'

    file_pattern = os.path.join(data_dir, f'{file_prefix}{partition}*.tfrecords')
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
            'isCoding': tf.io.FixedLenFeature([data_sz[5]], dtype=tf.int64),
            'isCorrect': tf.io.FixedLenFeature([data_sz[6]], dtype=tf.int64),
        }

        record_features = tf.io.parse_single_example(raw_record, feature_map)

        sequence = record_features['sequence']
        geneLength = record_features['geneLength']
        orfLength = record_features['orfLength']
        genomeGC = record_features['genomeGC']
        contigGC = record_features['contigGC']
        isCoding = tf.cast(record_features['isCoding'], dtype=tf.float32)
        isCorrect = tf.cast(record_features['isCorrect'], dtype=tf.float32)

        # return {'sequence':sequence, 'geneLength':geneLength, 'orfLength':orfLength, 'genomeGC':genomeGC, 'contigGC':contigGC}, (isCoding, isCorrect)
        return (sequence, geneLength, orfLength, genomeGC, contigGC), (isCoding, isCorrect)

    return process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, _parse_record_fn, num_epochs=epochs)

def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, parse_record_fn, num_epochs=None):
    """Given a Dataset with raw records, return an iterator over the records."""
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda raw_record: parse_record_fn(raw_record),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=True))

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset
