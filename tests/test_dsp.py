import librosa
import numpy as np
import ttslearn
from scipy.io import wavfile
from ttslearn.dsp import compute_delta, world_log_f0_vuv, world_spss_params


def test_world_spss():
    sr, x = wavfile.read(ttslearn.util.example_audio_file())
    x = x.astype(np.float64)
    feats = world_spss_params(x, sr, mgc_order=59)
    assert feats.shape[1] == 199

    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    feats_sr16 = world_spss_params(x, 16000)
    assert feats_sr16.shape[1] == 127

    # same number of frames
    assert feats.shape[0] == feats_sr16.shape[0]


def test_world_log_f0_vuv():
    sr, x = wavfile.read(ttslearn.util.example_audio_file())
    x = x.astype(np.float64)
    feats = world_log_f0_vuv(x, sr)
    assert feats.shape[1] == 4  # lf0 (static + delta + deltadelta) + vuv


def test_compute_delta():
    x = np.random.rand(10, 4)
    x_static = compute_delta(x, [1.0])
    assert x_static.shape == x.shape
    x_delta = compute_delta(x, [-0.5, 0.0, 0.5])
    assert x_delta.shape == x.shape
    x_deltadelta = compute_delta(x, [1.0, -2.0, 1.0])
    assert x_deltadelta.shape == x.shape
