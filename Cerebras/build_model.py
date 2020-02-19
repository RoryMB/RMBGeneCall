"""Build a Keras classification model."""

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Embedding, Conv1D, Add, Flatten
from tensorflow.keras.models import Model

def build_model(params, inputs=None):
    """Build a Keras classification NN."""

    dropout = params['dropout']

    # TODO: More inputs (gc, etc), another output (in CDS)
    in1 = Input(tensor=inputs['sequence'])
    in2 = Input(tensor=inputs['geneLength'])
    in3 = Input(tensor=inputs['orfLength'])
    in4 = Input(tensor=inputs['genomeGC'])
    in5 = Input(tensor=inputs['contigGC'])

    # There are 18 possible nucleotides (including ambiguity chars)
    # The input sequence is 183 bp
    # TODO: Pass this in params?
    x = Embedding(18, 4, input_length=183)(in1)

    x = Conv1D(256, kernel_size=3, strides=3, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(dropout)(x)

    # Residual / skip connections speed up convergence
    xSkip = Conv1D(32, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

    x = Conv1D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(dropout)(x)

    x = Conv1D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(dropout)(x)

    x = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = Add()([x, xSkip])
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(dropout)(x)

    x = Flatten()(x)

    x = Dense(128, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dense(32, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    out1 = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    out2 = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

    inputs = [in1, in2, in3, in4, in5]
    outputs = [out1, out2]

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
