import numpy as np
import torch
from tqdm import tqdm
from ttslearn.dsp import inv_mulaw_quantize, logmelspectrogram_to_audio
from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence


@torch.no_grad()
def synthesis_griffin_lim(device, sample_rate, labels, acoustic_model, scaler):
    """Synthesize waveform with Griffin-Lim algorithm.

    Args:
        device (torch.device): device to use for computation (CPU or GPU).
        sample_rate (int): sample rate of the output waveform.
        labels (list): list of labels to generate.
        acoustic_model (ttslearn.tacotron.models.Tacotron): acoustic model.
        scaler (sklearn.preprocessing.StandardScaler): preprocessing scaler.

    Returns:
        (torch.Tensor): waveform.
    """
    in_feats = text_to_sequence(pp_symbols(labels.contexts))
    in_feats = torch.tensor(in_feats, dtype=torch.long).to(device)

    # (T, C)
    _, out_feats, _, _ = acoustic_model.inference(in_feats)

    out_feats = out_feats.cpu().data.numpy()

    # Denormalization
    logmel = scaler.inverse_transform(out_feats)

    gen_wav = logmelspectrogram_to_audio(logmel, sample_rate)

    return gen_wav


@torch.no_grad()
def synthesis(device, sample_rate, labels, acoustic_model, wavenet_model, _tqdm=tqdm):
    """Synthesize waveform

    Args:
        device (torch.device): device to use for computation (CPU or GPU).
        sample_rate (int): sample rate of the output waveform.
        labels (list): list of labels to generate.
        acoustic_model (ttslearn.tacotron.models.Tacotron): acoustic model.
        wavenet_model (ttslearn.wavenet.WaveNet): WaveNet vocoder.
        _tqdm (optional): tqdm progress bar.

    Returns:
        (torch.Tensor): waveform.
    """
    in_feats = text_to_sequence(pp_symbols(labels.contexts))
    in_feats = torch.tensor(in_feats, dtype=torch.long).to(device)

    # (T, C)
    _, out_feats, _, _ = acoustic_model.inference(in_feats)

    # (B, T, C) -> (B, C, T)
    c = out_feats.view(1, -1, out_feats.size(-1)).transpose(1, 2)

    # 音声波形の長さを計算
    upsample_scale = np.prod(wavenet_model.upsample_scales)
    time_steps = (c.shape[-1] - wavenet_model.aux_context_window * 2) * upsample_scale

    # WaveNetによる音声波形の生成
    # NOTE: 計算に時間がかかるため、tqdmによるプログレスバーを受け付けるようにしている
    gen_wav = wavenet_model.inference(c, time_steps, _tqdm)

    # One-hotベクトルから一次元の信号に変換
    gen_wav = gen_wav.max(1)[1].float().cpu().numpy().reshape(-1)

    # Mu-law量子化の逆変換
    # NOTE: muは出力チャンネル数-1だと仮定
    gen_wav = inv_mulaw_quantize(gen_wav, wavenet_model.out_channels - 1)

    return gen_wav
