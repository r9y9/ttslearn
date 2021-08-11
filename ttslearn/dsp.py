import librosa
import numpy as np
import pysptk
import pyworld
from nnmnkwii.preprocessing import delta_features
from nnmnkwii.preprocessing.f0 import interp1d


def f0_to_lf0(f0):
    """Convert F0 to log-F0

    Args:
        f0 (ndarray): F0 in Hz.

    Returns:
        ndarray: log-F0.
    """
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    return lf0


def lf0_to_f0(lf0, vuv):
    """Convert log-F0 (and V/UV) to F0

    Args:
        lf0 (ndarray): F0 in Hz.
        vuv (ndarray): V/UV.

    Returns:
        ndarray: F0 in Hz.
    """
    f0 = np.exp(lf0)
    f0[vuv < 0.5] = 0
    return f0


def compute_delta(x, coef):
    """Compute delta features

    Args:
        x (ndarray): Feature vector.
        coef (ndarray): Coefficients.

    Returns:
        ndarray: Delta features.
    """
    y = np.zeros_like(x)
    # 特徴量の次元ごとに動的特徴量を計算
    for d in range(x.shape[1]):
        y[:, d] = np.correlate(x[:, d], coef, mode="same")
    return y


def world_log_f0_vuv(x, sr):
    """WORLD-based log-F0 and V/UV extraction

    Args:
        x (numpy.ndarray): Waveform.
        sr (int): Sampling rate.

    Returns:
        ndarray: Log-F0 and V/UV.
    """
    f0, timeaxis = pyworld.dio(x, sr)
    # (Optinal) Stonemask によってF0の推定結果をrefineする
    f0 = pyworld.stonemask(x, f0, timeaxis, sr)
    vuv = (f0 > 0).astype(np.float32)

    # 連続対数基本周波数
    lf0 = f0_to_lf0(f0)
    lf0 = interp1d(lf0)

    # 連続対数基本周波数と有声/無声フラグを2次元の行列の形にしておく
    lf0 = lf0[:, np.newaxis] if len(lf0.shape) == 1 else lf0
    vuv = vuv[:, np.newaxis] if len(vuv.shape) == 1 else vuv

    # 動的特徴量の計算
    windows = [
        [1.0],  # 静的特徴量に対する窓
        [-0.5, 0.0, 0.5],  # 1次動的特徴量に対する窓
        [1.0, -2.0, 1.0],  # 2次動的特徴量に対する窓
    ]
    lf0 = delta_features(lf0, windows)

    # すべての特徴量を結合
    feats = np.hstack([lf0, vuv]).astype(np.float32)

    return feats


def world_spss_params(x, sr, mgc_order=None):
    """WORLD-based acoustic feature extraction

    Args:
        x (ndarray): Waveform.
        sr (int): Sampling rate.
        mgc_order (int, optional): MGC order. Defaults to None.

    Returns:
        ndarray: WORLD features.
    """
    f0, timeaxis = pyworld.dio(x, sr)
    # (Optinal) Stonemask によってF0の推定結果をrefineする
    f0 = pyworld.stonemask(x, f0, timeaxis, sr)

    sp = pyworld.cheaptrick(x, f0, timeaxis, sr)
    ap = pyworld.d4c(x, f0, timeaxis, sr)

    alpha = pysptk.util.mcepalpha(sr)
    # メルケプストラムの次元数（※過去の論文にならい、16kHzの際に
    # 次元数が40（mgc_order + 1）になるように設定する
    # ただし、上限を 60 (59 + 1) とします
    # [Zen 2013] Statistical parametric speech synthesis using deep neural networks
    if mgc_order is None:
        mgc_order = min(int(sr / 16000.0 * 40) - 1, 59)
    mgc = pysptk.sp2mc(sp, mgc_order, alpha)

    # 有声/無声フラグ
    vuv = (f0 > 0).astype(np.float32)

    # 連続対数F0
    lf0 = f0_to_lf0(f0)
    lf0 = interp1d(lf0)
    # 帯域非周期性指標
    bap = pyworld.code_aperiodicity(ap, sr)

    # F0とvuvを二次元の行列の形にしておく
    lf0 = lf0[:, np.newaxis] if len(lf0.shape) == 1 else lf0
    vuv = vuv[:, np.newaxis] if len(vuv.shape) == 1 else vuv

    # 動的特徴量の計算
    windows = [
        [1.0],  # 静的特徴量に対する窓
        [-0.5, 0.0, 0.5],  # 1次動的特徴量に対する窓
        [1.0, -2.0, 1.0],  # 2次動的特徴量に対する窓
    ]
    mgc = delta_features(mgc, windows)
    lf0 = delta_features(lf0, windows)
    bap = delta_features(bap, windows)

    feats = np.hstack([mgc, lf0, vuv, bap]).astype(np.float32)

    return feats


def mulaw(x, mu=255):
    """Mu-Law companding.

    Args:
        x (ndarray): Input signal.
        mu (int): Mu.

    Returns:
        ndarray: Compressed signal.
    """
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def quantize(y, mu=255, offset=1):
    """Quantize the signal

    Args:
        y (ndarray): Input signal.
        mu (int): Mu.
        offset (int): Offset.

    Returns:
        ndarray: Quantized signal.
    """
    # [-1, 1] -> [0, 2] -> [0, 1] -> [0, mu]
    return ((y + offset) / 2 * mu).astype(np.int64)


def mulaw_quantize(x, mu=255):
    """Mu-law-quantize signal.

    Args:
        x (ndarray): Input signal.
        mu (int): Mu.

    Returns:
        ndarray: Quantized signal.
    """
    return quantize(mulaw(x, mu), mu)


def inv_mulaw(y, mu=255):
    """Inverse transformation of mu-law companding

    Args:
        y (ndarray): Input signal.
        mu (int): Mu.

    Returns:
        ndarray: Uncompressed signal.
    """
    return np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)


def inv_quantize(y, mu):
    """De-quantization.

    Args:
        y (ndarray): Input signal.
        mu (int): Mu.

    Returns:
        ndarray: Unquantized signal.
    """
    # [0, mu] -> [-1, 1]
    return 2 * y.astype(np.float32) / mu - 1


def inv_mulaw_quantize(y, mu=255):
    """Inverse transformation of mu-law quantization.

    Args:
        y (ndarray): Input signal.
        mu (int): Mu.

    Returns:
        ndarray: Unquantized signal.
    """
    return inv_mulaw(inv_quantize(y, mu), mu)


def logspectrogram(
    y,
    sr,
    n_fft=None,
    hop_length=None,
    win_length=None,
    clip=0.001,
):
    """Compute log-spectrogram.

    Args:
        y (ndarray): Waveform.
        sr (int): Sampling rate.
        n_fft (int, optional): FFT size.
        hop_length (int, optional): Hop length. Defaults to 12.5ms.
        win_length (int, optional): Window length. Defaults to 50 ms.
        clip (float, optional): Clip the magnitude. Defaults to 0.001.

    Returns:
        numpy.ndarray: Log-spectrogram.
    """
    if hop_length is None:
        hop_length = int(sr * 0.0125)
    if win_length is None:
        win_length = int(sr * 0.050)
    if n_fft is None:
        n_fft = next_power_of_2(win_length)

    S = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hanning"
    )
    # スペクトログラムのクリッピング
    # NOTE: クリッピングの値は、データに依存して調整する必要があります。
    # Tacotron 2の論文では 0.01 です
    S = np.maximum(np.abs(S), clip)

    # 対数を取る
    S = np.log10(S)

    # Time first: (T, N)
    return S.T


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def logmelspectrogram(
    y,
    sr,
    n_fft=None,
    hop_length=None,
    win_length=None,
    n_mels=80,
    fmin=None,
    fmax=None,
    clip=0.001,
):
    """Compute log-melspectrogram.

    Args:
        y (ndarray): Waveform.
        sr (int): Sampling rate.
        n_fft (int, optional): FFT size.
        hop_length (int, optional): Hop length. Defaults to 12.5ms.
        win_length (int, optional): Window length. Defaults to 50 ms.
        n_mels (int, optional): Number of mel bins. Defaults to 80.
        fmin (int, optional): Minimum frequency. Defaults to 0.
        fmax (int, optional): Maximum frequency. Defaults to sr / 2.
        clip (float, optional): Clip the magnitude. Defaults to 0.001.

    Returns:
        numpy.ndarray: Log-melspectrogram.
    """
    if hop_length is None:
        hop_length = int(sr * 0.0125)
    if win_length is None:
        win_length = int(sr * 0.050)
    if n_fft is None:
        n_fft = next_power_of_2(win_length)

    S = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hanning"
    )

    fmin = 0 if fmin is None else fmin
    fmax = sr // 2 if fmax is None else fmax

    # メルフィルタバンク
    mel_basis = librosa.filters.mel(sr, n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels)
    # スペクトログラム -> メルスペクトログラム
    S = np.dot(mel_basis, np.abs(S))

    # クリッピング
    S = np.maximum(S, clip)

    # 対数を取る
    S = np.log10(S)

    # Time first: (T, N)
    return S.T


def logmelspectrogram_to_audio(
    logmel,
    sr,
    n_fft=None,
    hop_length=None,
    win_length=None,
    fmin=None,
    fmax=None,
    n_iter=4,
):
    """Log-melspectrogram to audio.

    Args:
        logmel (ndarray): Log-melspectrogram.
        sr (int): Sampling rate.
        n_fft (int, optional): FFT size.
        hop_length (int, optional): Hop length. Defaults to 12.5ms.
        win_length (int, optional): Window length. Defaults to 50 ms.
        fmin (int, optional): Minimum frequency. Defaults to 0.
        fmax (int, optional): Maximum frequency. Defaults to sr / 2.
        n_iter (int, optional): Number of power iterations. Defaults to 4.

    Returns:
        numpy.ndarray: Waveform.
    """
    if hop_length is None:
        hop_length = int(sr * 0.0125)
    if win_length is None:
        win_length = int(sr * 0.050)
    if n_fft is None:
        n_fft = next_power_of_2(win_length)

    fmin = 0 if fmin is None else fmin
    fmax = sr // 2 if fmax is None else fmax

    mel = np.exp(logmel * np.log(10)).T
    S = librosa.feature.inverse.mel_to_stft(
        mel,
        n_fft=n_fft,
        power=1.0,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
    )
    y = librosa.griffinlim(
        S, hop_length=hop_length, win_length=win_length, window="hanning", n_iter=n_iter
    )

    return y
