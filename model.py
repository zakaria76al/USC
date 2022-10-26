import tensorflow as tf
from keras.layers import concatenate, Flatten, Dense, ConvLSTM2D, Dropout, Conv3D
from keras.models import Model
from keras.layers import Input, BatchNormalization, Reshape
from keras.callbacks import ModelCheckpoint


def conv_layer(input_layer, kernel, stride):
    output_layer = Conv3D(1, kernel_size=kernel, strides=stride)(input_layer)
    return output_layer


def tower(input_tower):
    output = Reshape((input_tower.shape[1], 1, input_tower.shape[2], 1))(input_tower)
    output = conv_layer(input_layer=output, kernel=(1, 1, 100), stride=(1, 1, 10))
    output = conv_layer(input_layer=output, kernel=(1, 1, 50), stride=(1, 1, 5))
    output = conv_layer(input_layer=output, kernel=(1, 1, 20), stride=(1, 1, 3))
    output = conv_layer(input_layer=output, kernel=(1, 1, 5), stride=(1, 1, 2))
    return output


def conv_bloc(input_conv_bloc, towers_nbr=40):
    output = tower(input_conv_bloc)
    for i in range(1, towers_nbr):
        output = concatenate([output, tower(input_conv_bloc)], axis=2)
    return output


def spatio_temporal_layer(input_st_layer, filters, kernel_size, return_sequences, use_dropout=True):
    output = ConvLSTM2D(filters, kernel_size=kernel_size, activation='relu', return_sequences=return_sequences,
                        data_format='channels_last')(input_st_layer)
    if use_dropout:
        output = (Dropout(0.5))(output)
    return output


def spatio_temporal_bloc(input_st_bloc, nb_classes, filter1, filter2, filter3):
    output = spatio_temporal_layer(multi_head, filters=filter1, kernel_size=(3, 3), return_sequences=True)
    output = spatio_temporal_layer(output, filters=filter2, kernel_size=(3, 3), return_sequences=True)
    output = spatio_temporal_layer(output, filters=filter3, kernel_size=(3, 3), return_sequences=False)
    output = (Flatten())(output)
    output = (Dense(200, activation='relu'))(output)
    output = (Dropout(0.5))(output)
    output = (Dense(100, activation='relu'))(output)
    output = (Dense(nb_classes, activation='softmax'))(output)
    return output


def input_model(input_shape):
    inputModel = Input(shape=input_shape)
    return inputModel


def end_to_end_architecture(model_input, params):
    output = BatchNormalization()(model_input)
    output = tf.keras.layers.Reshape((model_input.shape[1], model_input.shape[2], 1))(output)
    output = conv_bloc(output, params.tower_nbr)
    output = spatio_temporal_bloc(output, params.nb_classes, params.filter1, params.filter2, params.filter3)
    model = Model(inputs=model_input, outputs=output)
    return model


def mfcc_architecture(model_input, params):
    output = BatchNormalization()(model_input)
    output = tf.keras.layers.Reshape((model_input.shape[1], model_input.shape[2], model_input.shape[3], 1))(output)
    output = spatio_temporal_bloc(output, params.nb_classes, params.filter1, params.filter2, params.filter3)
    model = Model(inputs=model_input, outputs=output)
    return model


def create_model(model_input, model_output):
    model = Model(inputs=model_input, outputs=model_output)
    return model


def train_model(model, x_train, y_train, x_val, y_val, checkpoint_path, epochs, batch_size):
    checkpointer = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=True)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[checkpointer], verbose=1)
    return history
