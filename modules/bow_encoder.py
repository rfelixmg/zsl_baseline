def bow_encoder(input_size, hidden_layer, dropout=0):

    from keras.layers import Input, Dense, Dropout
    from keras.models import Model

    input_tensor = Input(shape=(input_size,), name='bag of words')

    encoded = Dropout(p=dropout)(input_tensor)
    encoded = Dense(hidden_layer, activation='relu')(encoded)

    encoder = Model(input=input_tensor, output=encoded)
    encoder.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy')

    return encoder