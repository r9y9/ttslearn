import numpy as np
import torch
from nnmnkwii.frontend import merlin as fe
from tqdm import tqdm
from ttslearn.dsp import inv_mulaw_quantize


@torch.no_grad()
def gen_waveform(
    device,  # cpu or cuda
    labels,  # フルコンテキストラベル
    logf0_vuv,  # 連続対数基本周波数と有声 / 無声フラグ
    wavenet_model,  # 学習済み WaveNet
    wavenet_in_scaler,  # 条件付け特徴量の正規化用 StandardScaler
    binary_dict,  # 二値特徴量を抽出する正規表現
    numeric_dict,  # 数値特徴量を抽出する正規表現
    tqdm=tqdm,  # プログレスバー
):
    """Generate waveform from WaveNet.

    Args:
        device (torch.device): torch.device to use.
        labels (nnmnkwii.io.hts.HTSLabel): full context labels.
        logf0_vuv (torch.Tensor): Log-f0 and V/UV flag.
        wavenet_model (nn.Module): Trained WaveNet.
        wavenet_in_scaler (sklearn.preprocessing.StandardScaler):
            StandardScaler for WaveNet input.
        binary_dict (dict): Dictionary of binary features.
        numeric_dict (dict): Dictionary of numeric features.
        tqdm (tqdm): tqdm progress bar.

    Returns:
        numpy.ndarray: Generated waveform.
    """
    # フレーム単位の言語特徴量の抽出
    in_feats = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        subphone_features="coarse_coding",
    )
    # フレーム単位の言語特徴量と、対数連続基本周波数・有声/無声フラグを結合
    in_feats = np.hstack([in_feats, logf0_vuv])

    # 特徴量の正規化
    in_feats = wavenet_in_scaler.transform(in_feats)

    # 条件付け特徴量を numpy.ndarray から torch.Tensor に変換
    c = torch.from_numpy(in_feats).float().to(device)
    # (B, T, C) -> (B, C, T)
    c = c.view(1, -1, c.size(-1)).transpose(1, 2)

    # 音声波形の長さを計算
    upsample_scale = np.prod(wavenet_model.upsample_scales)
    time_steps = (c.shape[-1] - wavenet_model.aux_context_window * 2) * upsample_scale

    # WaveNet による音声波形の生成
    # NOTE: 計算に時間を要するため、tqdm によるプログレスバーを利用します
    gen_wav = wavenet_model.inference(c, time_steps, tqdm)

    # One-hot ベクトルから1次元の信号に変換
    gen_wav = gen_wav.max(1)[1].float().cpu().numpy().reshape(-1)

    # Mu-law 量子化の逆変換
    gen_wav = inv_mulaw_quantize(gen_wav, wavenet_model.out_channels - 1)

    return gen_wav
