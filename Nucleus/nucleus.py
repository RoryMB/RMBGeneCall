#!/usr/bin/env python3

"""Nucleus compatible implementation of GeneCall model."""

import argparse
import numpy as np
import os
from bioutils import *
from datatoolkit import to_sparse_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Add, concatenate
from keras.layers import Conv1D
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Input, Flatten, Activation
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from kerastoolkit import tf_nowarn, select_gpu
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils import *

# TODO: Clean up imports

def build_dataset(args):
    print('Loading genomes...')
    genomes = read_genomes_from_list(genomeDir='genomes_proks/', genomeList=os.path.join(args.dataDirectory, 'genomes_train.tbl'))

    print('Analyzing orfs...')
    for genome in genomes:
        print('  Doing %s' % genome)
        for contig in genome.contigs:
            contig.find_orfs()
            realFeatures = [f for f in contig.features if f.featureType == 'CDS']
            contig.mark_coding_orfs(realFeatures)

    if args.dataset == 'all' or args.dataset == 'start':
        print('Building starts...')
        build_dataset_start(genomes, 90, 90, args)

    if args.dataset == 'all' or args.dataset == 'stop':
        print('Building stops...')
        raise NotImplementedError('Stop')
        # build_dataset_stop(genomes, 90, 90)

    if args.dataset == 'all' or args.dataset == 'coding':
        print('Building codings...')
        raise NotImplementedError('Coding')
        # build_dataset_coding(genomes, 33)

    if args.dataset == 'all' or args.dataset == 'score':
        print('Building scores...')
        raise NotImplementedError('Score')
        # build_dataset_score(genomes)

def build_dataset_start(genomes, sideLeft, sideRight, args):
    # Model input
    sequenceL = []
    geneLengthL = []
    orfLengthL = []
    genomeGCL = []
    contigGCL = []

    # Model output
    isCodingL = []
    isCorrectL = []

    # Metadata
    metaL = []

    for genome in genomes:
        print('  Doing %s' % genome)
        genomeGC = genome.gc
        for contig in genome.contigs:
            contigGC = contig.gc

            # Converting the DNA to a numerical representation eases storage
            # and requires less processing before submitting to a model
            dna = to_sparse_categorical(list(contig.dnaPos), encoder=dnaEncoder)

            orfsPos = [o for o in contig.orfs if o.strand == '+']
            orfsNeg = [o for o in contig.orfs if o.strand == '-']

            # TODO: Do orfsNeg
            for orf in orfsPos:
                for startPos in orf.starts:
                    # If the left side of the requested region runs off the edge of a contig,
                    # add the appropriate number of gap characters
                    l = startPos-sideLeft
                    if l < 0:
                        # TODO: Change [0] to equivalent in dnaEncoder['-']
                        extraL = [0] * (-l)
                        l = 0
                    else:
                        extraL = []

                    # If the right side of the requested region runs off the edge of a contig,
                    # add the appropriate number of gap characters
                    r = startPos+3+sideRight
                    if r > len(contig.dnaPos):
                        # TODO: Change [0] to equivalent in dnaEncoder['-']
                        extraR = [0] * (r-len(contig.dnaPos))
                        r = len(contig.dnaPos)
                    else:
                        extraR = []

                    sequence = np.array(np.concatenate((extraL, dna[l:r], extraR)), dtype=int)

                    geneLength = orf.right-startPos
                    orfLength = orf.right-orf.left
                    isCoding = True if orf.realStart else False
                    isCorrect = True if orf.realStart==startPos else False
                    # meta =

                    sequenceL.append(sequence)
                    geneLengthL.append(geneLength)
                    orfLengthL.append(orfLength)
                    genomeGCL.append(genomeGC)
                    contigGCL.append(contigGC)
                    isCodingL.append(isCoding)
                    isCorrectL.append(isCorrect)
                    # metaL.append(meta)

    print('  Saving files...')
    np.save(os.path.join(args.dataDirectory, 'start_sequence.npy'), sequenceL)
    np.save(os.path.join(args.dataDirectory, 'start_geneLength.npy'), geneLengthL)
    np.save(os.path.join(args.dataDirectory, 'start_orfLength.npy'), orfLengthL)
    np.save(os.path.join(args.dataDirectory, 'start_genomeGC.npy'), genomeGCL)
    np.save(os.path.join(args.dataDirectory, 'start_contigGC.npy'), contigGCL)
    np.save(os.path.join(args.dataDirectory, 'start_isCoding.npy'), isCodingL)
    np.save(os.path.join(args.dataDirectory, 'start_isCorrect.npy'), isCorrectL)
    # np.save(os.path.join(args.dataDirectory, 'start_meta.npy'), metaL)


def train(args):
    if args.dataset == 'all' or args.dataset == 'start':
        print('Training starts...')
        train_start(args)

    if args.dataset == 'all' or args.dataset == 'stop':
        print('Training stops...')
        raise NotImplementedError('Stop')

    if args.dataset == 'all' or args.dataset == 'coding':
        print('Training codings...')
        raise NotImplementedError('Coding')

    if args.dataset == 'all' or args.dataset == 'score':
        print('Training scores...')
        raise NotImplementedError('Score')

def train_start(args):
    print('Loading data...')
    sequence_trn = np.load(os.path.join(args.dataDirectory, 'start_sequence.npy'))
    geneLength_trn = np.load(os.path.join(args.dataDirectory, 'start_geneLength.npy'))
    orfLength_trn = np.load(os.path.join(args.dataDirectory, 'start_orfLength.npy'))
    genomeGC_trn = np.load(os.path.join(args.dataDirectory, 'start_genomeGC.npy'))
    contigGC_trn = np.load(os.path.join(args.dataDirectory, 'start_contigGC.npy'))
    isCoding_trn = np.load(os.path.join(args.dataDirectory, 'start_isCoding.npy'))
    isCorrect_trn = np.load(os.path.join(args.dataDirectory, 'start_isCorrect.npy'))

    print('Shuffling data...')
    # Shuffle and split dataset into training, validation, and test
    sequence_trn, sequence_tst, geneLength_trn, geneLength_tst, orfLength_trn, orfLength_tst, genomeGC_trn, genomeGC_tst, contigGC_trn, contigGC_tst, isCoding_trn, isCoding_tst, isCorrect_trn, isCorrect_tst = train_test_split(sequence_trn, geneLength_trn, orfLength_trn, genomeGC_trn, contigGC_trn, isCoding_trn, isCorrect_trn, random_state=7, test_size=0.2)
    # sequence_trn, sequence_val, geneLength_trn, geneLength_val, orfLength_trn, orfLength_val, genomeGC_trn, genomeGC_val, contigGC_trn, contigGC_val, isCoding_trn, isCoding_val, isCorrect_trn, isCorrect_val = train_test_split(sequence_trn, geneLength_trn, orfLength_trn, genomeGC_trn, contigGC_trn, isCoding_trn, isCorrect_trn, random_state=7, test_size=0.25)
    # (trnIn, trnOut), (valIn, valOut), (tstIn, tstOut) = tvt_split(samples, labels, valSplit=0.05, tstSplit=0.05)
    print(f'{len(sequence_trn):,} Trn | {len(sequence_tst):,} Tst')
    # print(f'{len(sequence_trn):,} Trn | {len(sequence_val):,} Val | {len(sequence_tst):,} Tst')

    print('Building model...')
    print('DBG: sequence_trn.shape', sequence_trn.shape)
    # Define model
    in1 = Input(shape=sequence_trn[0].shape)
    in2 = Input(shape=(1,))
    in3 = Input(shape=(1,))
    in4 = Input(shape=(1,))
    in5 = Input(shape=(1,))

    x = in1
    x = Embedding(len(dnaAll), 4, input_length=sequence_trn[0].shape)(x)

    x = Conv1D(256, kernel_size=3, strides=3, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    xSkip = Conv1D(32, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

    x = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Add()([x, xSkip])
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    x = concatenate([x, in2, in3, in4, in5], axis=-1)

    x = Dense(128, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    out1 = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    out2 = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

    inputs = [in1, in2, in3, in4, in5]
    outputs = [out1, out2]

    print('Compiling model...')
    # Compile and train model
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    print('Training model...')
    model.fit(
        [sequence_trn, geneLength_trn, orfLength_trn, genomeGC_trn, contigGC_trn],
        [isCoding_trn, isCorrect_trn],
        # validation_data=(valIn, valOut),
        validation_split=0.2,
        batch_size=4096,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)],
        epochs=1000,
        verbose=1,
    )

    print('Testing model...')
    tstPrdRaw = model.predict([sequence_tst, geneLength_tst, orfLength_tst, genomeGC_tst, contigGC_tst], batch_size=4096)
    tstPrd = np.around(tstPrdRaw)
    print('Coding ORF:')
    print_confusion(tstPrd[0], np.around(isCoding_tst))
    print('Correct start:')
    print_confusion(tstPrd[1], np.around(isCorrect_tst))

    print('Saving model...')
    model.save(os.path.join(args.dataDirectory, 'start.h5'))


def test(args):
    print('Loading genomes...')
    genomes = read_genomes_from_list(genomeDir='genomes_proks/', genomeList=os.path.join(args.dataDirectory, 'genomes_test.tbl'))

    print('Analyzing orfs...')
    for genome in genomes:
        print('  Doing %s' % genome)
        for contig in genome.contigs:
            contig.find_orfs()
            realFeatures = [f for f in contig.features if f.featureType == 'CDS']
            contig.mark_coding_orfs(realFeatures)

    if args.dataset == 'all' or args.dataset == 'start':
        print('Testing starts...')
        test_start(genomes, 90, 90, args)

    if args.dataset == 'all' or args.dataset == 'stop':
        print('Testing stops...')
        raise NotImplementedError('Stop')

    if args.dataset == 'all' or args.dataset == 'coding':
        print('Testing codings...')
        raise NotImplementedError('Coding')

    if args.dataset == 'all' or args.dataset == 'score':
        print('Testing scores...')
        raise NotImplementedError('Score')

def test_start(genomes, sideLeft, sideRight, args):
    # TODO: This function needlessly duplicates a bunch of build_dataset_start

    # Model input
    sequenceL = []
    geneLengthL = []
    orfLengthL = []
    genomeGCL = []
    contigGCL = []

    # Model output
    isCodingL = []
    isCorrectL = []

    # Metadata
    metaL = []

    for genome in genomes:
        print('  Doing %s' % genome)
        genomeGC = genome.gc
        for contig in genome.contigs:
            contigGC = contig.gc

            # Converting the DNA to a numerical representation eases storage
            # and requires less processing before submitting to a model
            dna = to_sparse_categorical(list(contig.dnaPos), encoder=dnaEncoder)

            orfsPos = [o for o in contig.orfs if o.strand == '+']
            orfsNeg = [o for o in contig.orfs if o.strand == '-']

            # TODO: Do orfsNeg
            for orf in orfsPos:
                for startPos in orf.starts:
                    # If the left side of the requested region runs off the edge of a contig,
                    # add the appropriate number of gap characters
                    l = startPos-sideLeft
                    if l < 0:
                        # TODO: Change [0] to equivalent in dnaEncoder['-']
                        extraL = [0] * (-l)
                        l = 0
                    else:
                        extraL = []

                    # If the right side of the requested region runs off the edge of a contig,
                    # add the appropriate number of gap characters
                    r = startPos+3+sideRight
                    if r > len(contig.dnaPos):
                        # TODO: Change [0] to equivalent in dnaEncoder['-']
                        extraR = [0] * (r-len(contig.dnaPos))
                        r = len(contig.dnaPos)
                    else:
                        extraR = []

                    sequence = np.array(np.concatenate((extraL, dna[l:r], extraR)), dtype=int)

                    geneLength = orf.right-startPos
                    orfLength = orf.right-orf.left
                    isCoding = True if orf.realStart else False
                    isCorrect = True if orf.realStart==startPos else False
                    meta = 'g=%s\tc=%s\to5=%d\to3=%d' % (genome.gid, contig.cid, startPos, orf.right)

                    sequenceL.append(sequence)
                    geneLengthL.append(geneLength)
                    orfLengthL.append(orfLength)
                    genomeGCL.append(genomeGC)
                    contigGCL.append(contigGC)
                    isCodingL.append(isCoding)
                    isCorrectL.append(isCorrect)
                    metaL.append(meta)

    sequence_tst = np.array(sequenceL)
    geneLength_tst = np.array(geneLengthL)
    orfLength_tst = np.array(orfLengthL)
    genomeGC_tst = np.array(genomeGCL)
    contigGC_tst = np.array(contigGCL)
    isCoding_tst = np.array(isCodingL)
    isCorrect_tst = np.array(isCorrectL)

    print('Loading model...')
    model = load_model('start.h5')

    print('Running model...')
    tstPrdRaw = model.predict([sequence_tst, geneLength_tst, orfLength_tst, genomeGC_tst, contigGC_tst], batch_size=4096)
    tstPrd = np.around(tstPrdRaw)
    print('Coding ORF:')
    print_confusion(tstPrd[0], np.around(isCoding_tst))
    print('Correct start:')
    print_confusion(tstPrd[1], np.around(isCorrect_tst))


def print_confusion(tstPrd, tstOut):
    """Prints a confusion matrix alongside class and overall accuracies."""

    conf = confusion_matrix(tstOut.flatten(), tstPrd.flatten())

    # The disaster that is fmtstr helps to make each row of the confusion matrix the same width
    # Look at the commented print below for a more interpretable version
    fmtstr = '%%%dd' % len(str(len(tstOut)))
    for c in range(conf.shape[0]):
        line = ' '.join([(fmtstr % i) for i in conf[c]])
        print('[', line, ']', '%.4f' % (conf[c][c]/sum(conf[c])))
        # print(conf[c], '%.4f' % (conf[c][c]/sum(conf[c])))

    corr = sum(tstOut.flatten()==tstPrd.flatten()) / tstOut.size
    print('Percent correct: %.4f' % corr)

    acc = (sum((conf[c][c]/sum(conf[c])) for c in range(conf.shape[0]))/2)
    print('Average class accuracy: %.4f' % acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='Comma separated list of GPUs to use')
    parser.add_argument('dataDirectory', type=str, help='Which directory the model and datasets are in')
    parser.add_argument('action', type=str, choices=['build', 'train', 'test'], help='What action to perform on the data')
    parser.add_argument('dataset', type=str, choices=['all', 'start', 'stop', 'coding', 'score'], help='Which data subset to manipulate')
    args = parser.parse_args()

    # Avoid the plethora of tensorflow debug messages
    tf_nowarn()
    select_gpu(args.gpu)

    if args.action == 'build':
        build_dataset(args)
    if args.action == 'train':
        train(args)
    if args.action == 'test':
        test(args)
