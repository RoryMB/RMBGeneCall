#!/usr/bin/env python3

"""TFRecord generation."""

import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

# def convert_to_tfr(x1, y2, path):
def convert_to_tfr(x1, x2, x3, x4, x5, y1, y2, path):
    """Write TFRecords from memory-resident features and label."""

    print("writing to {}".format(path))
    with tf.io.TFRecordWriter(path) as writer:
        for i in range(x1.shape[0]):
            # Sequence
            x1_curr = x1[i] # List
            x1_list = tf.train.FloatList(value=x1_curr)
            x1_feature = tf.train.Feature(float_list=x1_list)

            # Gene Length
            x2_curr = [x2[i]] # Scalar value
            x2_list = tf.train.FloatList(value=x2_curr)
            x2_feature = tf.train.Feature(float_list=x2_list)

            # Orf Length
            x3_curr = [x3[i]] # Scalar value
            x3_list = tf.train.FloatList(value=x3_curr)
            x3_feature = tf.train.Feature(float_list=x3_list)

            # Genome GC
            x4_curr = [x4[i]] # Scalar value
            x4_list = tf.train.FloatList(value=x4_curr)
            x4_feature = tf.train.Feature(float_list=x4_list)

            # Contig GC
            x5_curr = [x5[i]] # Scalar value
            x5_list = tf.train.FloatList(value=x5_curr)
            x5_feature = tf.train.Feature(float_list=x5_list)

            # Is Coding
            y1_curr = [y1[i]] # Scalar value
            y1_list = tf.train.Int64List(value=y1_curr)
            y1_feature = tf.train.Feature(int64_list=y1_list)

            # Is Correct
            y2_curr = [y2[i]] # Scalar value
            y2_list = tf.train.Int64List(value=y2_curr)
            y2_feature = tf.train.Feature(int64_list=y2_list)

            feature_dict = {
                # 'data': x1_feature,
                # 'label': y2_feature,
                'sequence': x1_feature,
                'geneLength': x2_feature,
                'orfLength': x3_feature,
                'genomeGC': x4_feature,
                'contigGC': x5_feature,
                'isCoding': y1_feature,
                'isCorrect': y2_feature,
            }
            feature_set = tf.train.Features(feature=feature_dict)
            example = tf.train.Example(features=feature_set)

            writer.write(example.SerializeToString())
            if i % 1000 == 0:
                sys.stdout.write("writing record {} of {}\r".format(i, x1.shape[0]))

        print("{} records written to {}".format(x1.shape[0], path))

def main(outpfx=None):
    """TFRecord generation."""

    # Load features and labels
    sequence = np.load('model1/start_sequence.npy')#[:2**20]
    geneLength = np.load('model1/start_geneLength.npy')#[:2**20]
    orfLength = np.load('model1/start_orfLength.npy')#[:2**20]
    genomeGC = np.load('model1/start_genomeGC.npy')#[:2**20]
    contigGC = np.load('model1/start_contigGC.npy')#[:2**20]
    isCoding = np.load('model1/start_isCoding.npy')#[:2**20]
    isCorrect = np.load('model1/start_isCorrect.npy')#[:2**20]

    # Split data into training and testing
    sequence_train, sequence_test = train_test_split(sequence, test_size=0.2, random_state=42)
    geneLength_train, geneLength_test = train_test_split(geneLength, test_size=0.2, random_state=42)
    orfLength_train, orfLength_test = train_test_split(orfLength, test_size=0.2, random_state=42)
    genomeGC_train, genomeGC_test = train_test_split(genomeGC, test_size=0.2, random_state=42)
    contigGC_train, contigGC_test = train_test_split(contigGC, test_size=0.2, random_state=42)
    isCoding_train, isCoding_test = train_test_split(isCoding, test_size=0.2, random_state=42)
    isCorrect_train, isCorrect_test = train_test_split(isCorrect, test_size=0.2, random_state=42)

    print(sequence_train.shape)
    print(geneLength_train.shape)
    print(orfLength_train.shape)
    print(genomeGC_train.shape)
    print(contigGC_train.shape)
    print(isCoding_train.shape)
    print(isCorrect_train.shape)

    if outpfx:
        outpfx += '-'

    train_name = outpfx + 'train.tfrecords'
    train_path = os.path.join(os.getcwd(), train_name)
    test_name = outpfx + 'test.tfrecords'
    test_path = os.path.join(os.getcwd(), test_name)

    # Write train and test tfrecord datasets
    convert_to_tfr(sequence_train, geneLength_train, orfLength_train, genomeGC_train, contigGC_train, isCoding_train, isCorrect_train, train_path)
    convert_to_tfr(sequence_test, geneLength_test, orfLength_test, genomeGC_test, contigGC_test, isCoding_test, isCorrect_test, test_path)
    # convert_to_tfr(sequence_train, isCorrect_train, train_path)
    # convert_to_tfr(sequence_test, isCorrect_test, test_path)

    print('TFRecords generated')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert CSV to TFRecords')

    parser.add_argument('--outpfx', type=str, default='', help='string prepended to output files')
    args = vars(parser.parse_args())
    main(**args)
