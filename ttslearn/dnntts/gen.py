import numpy as np
import pysptk
import pyworld
import torch
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter
from ttslearn.dnntts.multistream import (
    get_static_stream_sizes,
    get_windows,
    multi_stream_mlpg,
    split_streams,
)


@torch.no_grad()
def predict_duration(
    device,  # cpu or cuda
    labels,  # フルコンテキストラベル
    duration_model,  # 学習済み継続長モデル
    duration_config,  # 継続長モデルの設定
    duration_in_scaler,  # 言語特徴量の正規化用 StandardScaler
    duration_out_scaler,  # 音素継続長の正規化用 StandardScaler
    binary_dict,  # 二値特徴量を抽出する正規表現
    numeric_dict,  # 数値特徴量を抽出する正規表現
):
    """Predict phoneme durations.

    Args:
        device (torch.device): pytorch device
        labels (list): list of labels
        duration_model (nn.Module): trained duration model
        duration_config (dict): configuration of duration model
        duration_in_scaler (sklearn.preprocessing.StandardScaler):
            StandardScaler of duration features
        duration_out_scaler (sklearn.preprocessing.StandardScaler):
            StandardScaler of duration output
        binary_dict (dict): dictionary of binary features
        numeric_dict (dict): dictionary of numeric features

    Returns:
        numpy.ndarray: predicted durations
    """
    # 言語特徴量の抽出
    in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict).astype(
        np.float32
    )

    # 言語特徴量の正規化
    in_feats = duration_in_scaler.transform(in_feats)

    # 継続長の予測
    x = torch.from_numpy(in_feats).float().to(device).view(1, -1, in_feats.shape[-1])
    pred_durations = duration_model(x, [x.shape[1]]).squeeze(0).cpu().data.numpy()

    # 予測された継続長に対して、正規化の逆変換を行います
    pred_durations = duration_out_scaler.inverse_transform(pred_durations)

    # 閾値処理
    pred_durations[pred_durations <= 0] = 1
    pred_durations = np.round(pred_durations)

    return pred_durations


@torch.no_grad()
def predict_acoustic(
    device,  # CPU or GPU
    labels,  # フルコンテキストラベル
    acoustic_model,  # 学習済み音響モデル
    acoustic_config,  # 音響モデルの設定
    acoustic_in_scaler,  # 言語特徴量の正規化用 StandardScaler
    acoustic_out_scaler,  # 音響特徴量の正規化用 StandardScaler
    binary_dict,  # 二値特徴量を抽出する正規表現
    numeric_dict,  # 数値特徴量を抽出する正規表現
    mlpg=True,  # MLPG を使用するかどうか
):
    """Predict acoustic features.

    Args:
        device (torch.device): pytorch device
        labels (list): list of labels
        acoustic_model (nn.Module): trained acoustic model
        acoustic_config (dict): configuration of acoustic model
        acoustic_in_scaler (sklearn.preprocessing.StandardScaler):
            StandardScaler of acoustic features
        acoustic_out_scaler (sklearn.preprocessing.StandardScaler):
            StandardScaler of acoustic output
        binary_dict (dict): dictionary of binary features
        numeric_dict (dict): dictionary of numeric features
        mlpg (bool): whether to use MLPG

    Returns:
        numpy.ndarray: predicted acoustic features
    """
    # フレーム単位の言語特徴量の抽出
    in_feats = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        subphone_features="coarse_coding",
    )
    # 正規化
    in_feats = acoustic_in_scaler.transform(in_feats)

    # 音響特徴量の予測
    x = torch.from_numpy(in_feats).float().to(device).view(1, -1, in_feats.shape[-1])
    pred_acoustic = acoustic_model(x, [x.shape[1]]).squeeze(0).cpu().data.numpy()

    # 予測された音響特徴量に対して、正規化の逆変換を行います
    pred_acoustic = acoustic_out_scaler.inverse_transform(pred_acoustic)

    # パラメータ生成アルゴリズム (MLPG) の実行
    if mlpg and np.any(acoustic_config.has_dynamic_features):
        # (T, D_out) -> (T, static_dim)
        pred_acoustic = multi_stream_mlpg(
            pred_acoustic,
            acoustic_out_scaler.var_,
            get_windows(acoustic_config.num_windows),
            acoustic_config.stream_sizes,
            acoustic_config.has_dynamic_features,
        )

    return pred_acoustic


def gen_waveform(
    sample_rate,  # サンプリング周波数
    acoustic_features,  # 音響特徴量
    stream_sizes,  # ストリームサイズ
    has_dynamic_features,  # 音響特徴量が動的特徴量を含むかどうか
    num_windows=3,  # 動的特徴量の計算に使う窓数
    post_filter=False,  # フォルマント強調のポストフィルタを使うかどうか
):
    """Generate waveform from acoustic features.

    Args:
        sample_rate (int): sampling rate
        acoustic_features (numpy.ndarray): acoustic features
        stream_sizes (list): list of stream sizes
        has_dynamic_features (list): whether the acoustic features contains dynamic features
        num_windows (int): number of windows
        post_filter (bool): whether to use post filter

    Returns:
        numpy.ndarray: waveform
    """
    # 静的特徴量の次元数を取得
    if np.any(has_dynamic_features):
        static_stream_sizes = get_static_stream_sizes(
            stream_sizes, has_dynamic_features, num_windows
        )
    else:
        static_stream_sizes = stream_sizes

    # 結合された音響特徴量をストリーム毎に分離
    mgc, lf0, vuv, bap = split_streams(acoustic_features, static_stream_sizes)

    fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
    alpha = pysptk.util.mcepalpha(sample_rate)

    # フォルマント強調のポストフィルタ
    if post_filter:
        mgc = merlin_post_filter(mgc, alpha)

    # 音響特徴量を音声パラメータに変換
    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(
        bap.astype(np.float64), sample_rate, fftlen
    )
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

    # WORLD ボコーダを利用して音声生成
    gen_wav = pyworld.synthesize(
        f0.flatten().astype(np.float64),
        spectrogram.astype(np.float64),
        aperiodicity.astype(np.float64),
        sample_rate,
    )

    return gen_wav
