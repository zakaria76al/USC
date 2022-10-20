import librosa
from scipy.io import wavfile
import scipy.signal as sps
import numpy as np


def load_wav(file, sr, use_librosa=True):
    if use_librosa:
        wav, sr = librosa.load(file, sr=sr)
    else:
        sampling_rate, data = wavfile.read(file)
        number_of_samples = round(len(data) * float(sr) / sampling_rate)
        wav = sps.resample(data, number_of_samples)
    return wav


def sliding_windows(data, frame_size, hop_size):
    res = []
    for i in range(0, len(data) - hop_size, hop_size):
        if len(data) >= i + frame_size:
            res.append(data[i:i + frame_size])
    return res


def create_raw_sequences(data, params):
    frame_size_seq = (params.frame_size - params.hop_size) * (params.sequence_nbr - 1) + params.frame_size
    sequences = []
    for i in range(0, len(data) - params.hop_size, params.hop_size):
        if len(data) >= i + frame_size_seq:
            sequences.append(sliding_windows(data[i: i + frame_size_seq], params.frame_size, params.hop_size))
    return sequences


def create_mfcc_sequences(data, params):
    frame_size_seq = (params.frame_size - params.hop_size) * (params.sequence_nbr - 1) + params.frame_size
    sequences = []
    for i in range(0, len(data) - params.hop_size, params.hop_size):
        if len(data) >= i + frame_size_seq:
            windows = sliding_windows(data[i:i + frame_size_seq], params.frame_size, params.hop_size)
            mfccs_features = []
            for window in windows:
                mfccs_features.append(librosa.feature.mfcc(y=window, sr=params.sample_rate, n_mfcc=params.
                                                           mfcc_coefficients))
            sequences.append(mfccs_features)
    return sequences

