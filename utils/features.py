import librosa
import numpy as np

N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20


def load_audio(audio_path):
    """Load audio files

    Args:
        audio_path (str): Path to audio file

    Returns:
        y: Audio samples over time
        sr: Sampling rate
    """
    return librosa.load(path=audio_path, sr=None) 


def compress_features(features):
    """
    Summarizes all features into a 2xn vector where n is the number of features.
    This is to reduce the size of the feature vectors that we obtain from our clips.
    By doing this, we discard any temporal ordering.
    """
    features = np.asanyarray(features)
    mean = features.mean(axis=1)
    std = features.std(axis=1)
    return np.concatenate([mean, std])


def extract_mfcc(y, sr, *,
                 n_fft=N_FFT,
                 win_length=WIN_LENGTH,
                 hop_length=HOP_LENGTH,
                 n_mels=N_MELS,
                 n_mfcc=N_MFCC):
    """
    Extracts MFCC using librosa implementation.

    Args:
        y: Audio as ndarray.
        sr: Sampling rate.
        n_mfcc: Number of

    Returns:
        MFCCs: Array of n MFCCs.
    """
    return librosa.feature.mfcc(y=y, sr=sr,
                                n_fft=n_fft,
                                win_length=win_length,
                                hop_length=hop_length,
                                n_mels=n_mels,
                                n_mfcc=n_mfcc)


def extract_spectral_centroid(y, sr, *,
                              n_fft=N_FFT,
                              win_length=WIN_LENGTH,
                              hop_length=HOP_LENGTH,
                              ):
    """Extracts spectral centroid with Librosa.

    Args:
        y: Audio as ndarray.
        sr: Sampling rate.
        n_fft: STFT frame length
        win_length: Window function size.
        hop_length: STFT hop length. 

    Returns:
        centroids: The centroid freqs.
    """
    return librosa.feature.spectral_centroid(y=y,
                                             sr=sr,
                                             n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length)


def extract_log_mel_spectrogram(y, sr, *,
                                n_fft=N_FFT,
                                win_length=WIN_LENGTH,
                                hop_length=HOP_LENGTH,
                                n_mels=N_MELS,):

    return librosa.power_to_db(librosa.feature.melspectrogram(y=y,
                                          sr=sr,
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          win_length=win_length,
                                          n_mels=n_mels))


def extract_zero_crossing_rate(y, sr, *,
                               frame_length=WIN_LENGTH,
                               hop_length=HOP_LENGTH,
                               ):
    return librosa.feature.zero_crossing_rate(y=y,
                                              frame_length=frame_length,
                                              hop_length=hop_length)


def extract_chroma(y, sr, *,
                   n_fft=N_FFT,
                   win_length=WIN_LENGTH,
                   hop_length=HOP_LENGTH,
                   ):

    return librosa.feature.chroma_stft(y=y,
                                       sr=sr,
                                       n_fft=n_fft,
                                       win_length=win_length,
                                       hop_length=hop_length)
