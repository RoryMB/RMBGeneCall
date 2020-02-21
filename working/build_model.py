"""Build a Keras classification model."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Embedding, Conv1D, Add, Flatten, Reshape, concatenate
from tensorflow.keras.models import Model

def build_model(params, inputs=None):
    """Build a Keras classification NN."""

    dropout = params['dropout']

    in1 = Input(tensor=tf.reshape(inputs['sequence'], (-1, 183, 18)))
    # in2 = Input(tensor=inputs['geneLength'])
    # in3 = Input(tensor=inputs['orfLength'])
    # in4 = Input(tensor=inputs['genomeGC'])
    # in5 = Input(tensor=inputs['contigGC'])

    # There are 18 possible nucleotides (including ambiguity chars)
    # The input sequence is 183 bp
    # TODO: Pass this in params?
    # x = Embedding(18, 4, input_length=183)(in1)
    x = in1

    # x = Conv1D(256, kernel_size=3, strides=3, kernel_initializer='he_normal')(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.4)(x)

    # Residual / skip connections speed up convergence
    # xSkip = Conv1D(32, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

    # x = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    # x = Activation('relu')(x)
    # x = Dropout(dropout)(x) if dropout else x

    # x = Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    # x = Activation('relu')(x)
    # x = Dropout(dropout)(x) if dropout else x

    # x = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    # x = Add()([x, xSkip])
    # x = Activation('relu')(x)
    # x = Dropout(dropout)(x) if dropout else x

    x = Flatten()(x)

    # x = concatenate([x, in2, in3, in4, in5], axis=-1)

    x = Dense(128, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    out = Dense(2, activation=None, kernel_initializer='he_normal')(x)
    # out1 = Reshape(target_shape=(2, 1))(out1)
    # out2 = Dense(2, activation=None, kernel_initializer='he_normal')(x)
    # out2 = Reshape(target_shape=(2, 1))(out2)
    # out = concatenate([out, out], axis=-1)

    inputs = [in1]
    outputs = [out]

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
