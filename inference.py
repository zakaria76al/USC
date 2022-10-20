import params
import model
import features
import numpy as np


if __name__ == '__main__':
    print("USC welcome")
    #  ***** Data preparation *****
    # data = features.load_wav("file.wav", sr=16000, use_librosa=True)
    # p1 = params.params()
    # sequence = features.create_raw_sequences(data, params=p1)       # Verified
    # sequence = features.create_mfcc_sequences(data, params=p1)      # Verified
    # print(np.shape(sequence))

    # # ***** Model test *****
    p1 = params.params(nb_classes=4)

    # To create the end-to-end architecture uncomment the following lines
    # input_model = model.input_model(input_shape=(p1.sequence_nbr, p1.frame_size))
    # usc = model.end_to_end_architecture(model_input=input_model, params=p1)
    # usc.summary()  # Verified

    # To create the hybrid architecture uncomment the following lines
    input_model = model.input_model(input_shape=(p1.sequence_nbr, p1.mfcc_coefficients, 31))
    usc = model.mfcc_architecture(model_input=input_model, params=p1)
    usc.summary()     # Verified

    # Model training format, data(X_train, y_train, X_val, y_val) should be a numpy array.
    # checkpoint_path = "weights/usc_weight-{epoch:04d}.hdf5"
    # history = model.train_model(usc, X_train, y_train, X_val, y_val, checkpoint_path, epochs=20, batch_size=32)

