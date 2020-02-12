"""Build a Keras classification model."""

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Embedding, Conv1D, Add, Flatten
from tensorflow.keras.models import Model

def build_model(params, tensor=None):
    """Build a Keras classification NN."""

    dropout = params['dropout']

    # TODO: More inputs (gc, etc), another output (in CDS)
    inp = Input(tensor=tensor)

    # There are 18 possible nucleotides (including ambiguity chars)
    # The input sequence is 183 bp
    # TODO: Pass this in params?
    x = Embedding(18, 4, input_length=183)(inp)

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

    out = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

    inputs = inp
    outputs = out

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
