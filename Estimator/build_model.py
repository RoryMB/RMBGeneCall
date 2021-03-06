"""Build a Keras classification model."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Embedding, Dropout, concatenate, Conv1D, Conv2D, Add
from tensorflow.keras.layers import BatchNormalization, multiply
from tensorflow.keras.models import Model

def build_model_1to1_gpu(params, inputs=None):
    """Build a Keras classification NN."""

    dropout = params['dropout']

    # TODO: More inputs (gc, etc), another output (in CDS)

    # There are 18 possible nucleotides (including ambiguity chars)
    # The input sequence is 183 bp
    # TODO: Pass this in params?
    x = in1 = Input(tensor=tf.reshape(inputs['sequence'], (-1, 183, 18)))

    x = Conv1D(256, kernel_size=3, strides=3, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    # Residual / skip connections speed up convergence
    xSkip = Conv1D(32, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

    x = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Add()([x, xSkip])
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Flatten()(x)

    x = Dense(128, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    out = Dense(2, activation=None)(x)

    inputs = in1
    outputs = out

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

def build_model_1to1_cs1(params, inputs=None):
    """Build a Keras classification NN."""

    x = in1 = Input(tensor=tf.reshape(inputs['sequence'], (-1, 18, 183, 1)))
    x = Flatten()(x)
    x = Dense(3000, activation='relu')(x)
    out = Dense(2, activation=None)(x)

    inputs = in1
    outputs = out

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

def build_model_1to1_ipu(params, inputs=None):
    """Build a Keras classification NN."""

    dropout = params['dropout']

    x = in1 = Input(tensor=tf.reshape(inputs['sequence'], (-1, 183, 18)))

    x = Conv1D(256, kernel_size=3, strides=3, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Flatten()(x)

    x = Dense(128, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    out = Dense(2, activation=None)(x)

    inputs = in1
    outputs = out

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

def build_model_5to2_ipu(params, inputs=None):
    """Build a Keras classification NN."""

    dropout = 0.4

    x = in1 = Input(tensor=tf.reshape(inputs['sequence'], (-1, 183, 18)))
    in2 = Input(tensor=inputs['geneLength'])
    in3 = Input(tensor=inputs['orfLength'])
    in4 = Input(tensor=inputs['genomeGC'])
    in5 = Input(tensor=inputs['contigGC'])

    x = Conv1D(256, kernel_size=3, strides=3, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    #xSkip = Conv1D(32, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

    x = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    #x = Add()([x, xSkip])
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout else x

    x = Flatten()(x)

    x = concatenate([x, in2, in3, in4, in5], axis=-1)

    x = Dense(128, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    out1 = Dense(2, activation=None, kernel_initializer='he_normal')(x)
    out1 = Reshape(target_shape=(2, 1))(out1)
    out2 = Dense(2, activation=None, kernel_initializer='he_normal')(x)
    out2 = Reshape(target_shape=(2, 1))(out2)
    out = concatenate([out1, out2], axis=-1)

    inputs = [in1, in2, in3, in4, in5]
    outputs = [out]

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
