import librosa

N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20


def extract_mfcc(y, sr,
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


def extract_spectral_centroids(y, sr, ):
    pass


def extract_mel_spectrogram():
    pass


def extract_mel_spectrogram_db_magnitude():
    pass


def extract_zero_crossings():
    pass


def extract_chroma():
    pass
