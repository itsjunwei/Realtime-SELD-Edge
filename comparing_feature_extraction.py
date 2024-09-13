import librosa
import numpy as np
import time
from rich.progress import track
import gc
import math
import sys

gc.enable()
gc.collect()

# Global variables
NFFT = 512
FS = 24000
HOP_LEN = 300
WIN_LEN = 512
MEL_BINS = 128

# Comparison variables needed
N_ATTEMPTS = 500
N_SECS = 2
duration = int(N_SECS * FS)

# Mel Filter Variables
melW = librosa.filters.mel(sr=FS, n_fft=NFFT, n_mels=MEL_BINS, fmin=50, fmax=None)
salsa_W = melW.T

# Generic SALSA-Lite Variables
_c = 343
_delta = 2 * np.pi * FS / (NFFT * _c)
d_max = 42/1000
fmax_doa = 4000
fmin_doa = 50
fmax = 9000
n_bins = NFFT // 2 + 1
lower_bin = int(np.floor(fmin_doa * NFFT / float(FS))) # 1
upper_bin = int(np.floor(fmax_doa * NFFT / float(FS))) # 42
lower_bin = np.max((1, lower_bin))
cutoff_bin = int(np.floor(fmax * NFFT / float(FS)))
freq_vector = np.arange(n_bins)
freq_vector[0] = 1
freq_vector = freq_vector[:, None, None]  # n_bins x 1 x 1


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

def get_gccphat(audio_data):
    """
    Reimplementation of the MelSpec-GCCPHAT extraction process from the DCASE 2024 Challenge Task 3 Baseline code.

    Parameters
    ----------
    audio_data : np.array
        Audio data in numpy array of shape (4 x T), with 4 audio channels and T samples

    Returns
    --------
    melgcc : np.array
        MelSpec-GCCPHAT features
    """

    audio_in = audio_data.T # Reshape to T x 4
    _nb_ch = audio_in.shape[1]

    # Linear Spectrogram extraction
    spectra = []
    for ch_cnt in range(_nb_ch):
        stft_ch = librosa.core.stft(np.asfortranarray(audio_in[:, ch_cnt]), n_fft=NFFT, hop_length=HOP_LEN,
                                    win_length=WIN_LEN, window="hann")
        spectra.append(stft_ch)
    spectra = np.array(spectra).T


    # Mel Spectrogram extraction    
    mel_feat = np.zeros((spectra.shape[0], MEL_BINS, spectra.shape[-1]))
    for ch_cnt in range(spectra.shape[-1]):
        mag_spectra = np.abs(spectra[:, :, ch_cnt]) ** 2
        mel_spectra = np.dot(mag_spectra, melW.T)
        log_mel_spectra = librosa.power_to_db(mel_spectra)
        mel_feat[:, :, ch_cnt] = log_mel_spectra
    mel_feat = mel_feat.transpose((2, 0, 1))


    # GCC-PHAT Spatial feature extraction
    gcc_channels = nCr(spectra.shape[-1], 2)
    gcc_feat = np.zeros((spectra.shape[0], MEL_BINS, gcc_channels))
    cnt = 0
    for m in range(spectra.shape[-1]):
        for n in range(m+1, spectra.shape[-1]):
            R = np.conj(spectra[:, :, m]) * spectra[:, :, n]
            cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
            cc = np.concatenate((cc[:, -MEL_BINS//2:], cc[:, :MEL_BINS//2]), axis=-1)
            gcc_feat[:, :, cnt] = cc
            cnt += 1

    # Final MelSpec + GCC-PHAT feature
    gcc_feat = gcc_feat.transpose((2, 0, 1))
    gcc_feat = np.concatenate((mel_feat, gcc_feat), axis=0)

    return gcc_feat


def extract_salsalite(audio_data, use_mel=False):
    """Extracts SALSA-Lite features from an array of audio data in the shape of (n_channels, samples)
    Also has the option to use the SALSA-Mel variant by setting `use_mel` to be `True`

    Parameters
    ----------
    audio_data : array
        Array of audio data in the shape of (n_channels, n_samples)
    use_mel : Boolean (optional)
        True if we want to use SALSA-Mel variant

    Returns
    -------
    SALSA-Lite feature : array
        SALSA-Lite or SALSA-Mel features in the shape of (n_channels, n_timebins, n_freqbins)
    """
    n_mics = audio_data.shape[0]

    # Pre-allocate the array for STFT
    for imic in range(n_mics):
        stft = librosa.stft(y=np.asfortranarray(audio_data[imic]),
                            n_fft=NFFT,
                            hop_length=HOP_LEN,
                            center=True,
                            window='hann',
                            pad_mode='reflect')

        # Compute log linear power spectrum
        spec = np.abs(stft) ** 2
        if use_mel:
            spec = np.dot(spec.T, salsa_W).T
        log_spec = librosa.power_to_db(spec, ref=1.0, amin=1e-10, top_db=None)

        # Append to the arrays
        if imic == 0:
            log_specs = np.empty((n_mics, log_spec.shape[0], log_spec.shape[1]), dtype=np.float32)
            stft_result = np.empty((n_mics, stft.shape[0], stft.shape[1]), dtype=np.complex64)
        stft_result[imic] = stft
        log_specs[imic] = log_spec

    # Transpose into better shape
    X = np.transpose(stft_result, (1, 2, 0))
    log_specs = np.transpose(log_specs, (1, 2, 0))

    # Compute spatial feature
    phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))
    phase_vector = phase_vector / (_delta * freq_vector)

    if use_mel:
        phase_vector[upper_bin: , : , : ] = 0
        phase_vector = np.dot(phase_vector.T, salsa_W).T
    else:
        log_specs = log_specs[lower_bin:cutoff_bin , : , : ]
        phase_vector = phase_vector[lower_bin:cutoff_bin , : , : ]
        phase_vector[upper_bin: , : , : ] = 0

    # Stack features and return it 
    return np.concatenate((log_specs, phase_vector), axis=-1).T


if __name__ == "__main__":

    if len(sys.argv) == 2:
        feature_choice = sys.argv[1]
    else:
        feature_choice = "all"

    print("Comparing {}seconds feature extraction for {} times!".format(N_SECS, N_ATTEMPTS))
    print("\tFeatures compared : {}".format(feature_choice))


    if feature_choice == "gcc" or feature_choice == "all":

        melgcc = []
        for _ in track(range(N_ATTEMPTS), description="Extracting Melspec GCC-PHAT..."):
            random_input = np.random.rand(4, duration)

            gcc_start = time.time()
            _ = get_gccphat(random_input)
            melgcc.append(time.time() - gcc_start)

            time.sleep(0.1)

        # We remove the first few iterations in case of any startup latency
        melgcc = np.array(melgcc[10:])
        print("[MelGCC] Extraction Time ~ N({:0.3f}, {:0.3f})".format(np.mean(melgcc), np.var(melgcc)))

        gc.enable()
        gc.collect()


    if feature_choice == "salsalite" or feature_choice == "all":

        salsalite_time = []
        for _ in track(range(N_ATTEMPTS), description='Extracting SALSA-Lite...'):
            random_input = np.random.rand(4,duration)

            start_time = time.time()
            salsalite = extract_salsalite(random_input)
            salsalite_time.append(time.time()-start_time)

            time.sleep(0.1)

        # We remove the first few iterations in case of any startup latency
        salsalite_time = np.array(salsalite_time[10:])
        print("[SALSA-Lite] Extraction Time ~ N({:0.3f}, {:0.3f})".format(np.mean(salsalite_time), np.var(salsalite_time)))

        gc.enable()
        gc.collect()

    if feature_choice == "salsamel" or feature_choice == "all":

        salsamel_time = []
        for _ in track(range(N_ATTEMPTS), description='Extracting SALSA-Mel...'):
            random_input = np.random.rand(4,duration)

            start_time = time.time()
            salsa_mel_lite = extract_salsalite(random_input, use_mel=True)
            salsamel_time.append(time.time()-start_time)

            time.sleep(0.1)

        # We remove the first few iterations in case of any startup latency
        salsamel_time = np.array(salsamel_time[10:])
        print("[SALSA-Mel] Extraction Time ~ N({:0.3f}, {:0.3f})".format(np.mean(salsamel_time), np.var(salsamel_time)))