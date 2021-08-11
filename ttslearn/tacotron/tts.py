import json
from pathlib import Path

import numpy as np
import pyopenjtalk
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from ttslearn.dsp import inv_mulaw_quantize, logmelspectrogram_to_audio
from ttslearn.pretrained import retrieve_pretrained_model
from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence
from ttslearn.util import StandardScaler


class Tacotron2TTS(object):
    """Tacotron 2 based text-to-speech

    Args:
        model_dir (str): model directory. A pre-trained model (ID: ``tacotron2``)
            is used if None.
        device (str): cpu or cuda.

    Examples:

        >>> from ttslearn.tacotron import Tacotron2TTS
        >>> engine = Tacotron2TTS()
        >>> wav, sr = engine.tts("一貫学習にチャレンジしましょう！")
    """

    def __init__(self, model_dir=None, device="cpu"):
        self.device = device

        if model_dir is None:
            model_dir = retrieve_pretrained_model("tacotron2")
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # search for config.yaml
        if (model_dir / "config.yaml").exists():
            config = OmegaConf.load(model_dir / "config.yaml")
            self.sample_rate = config.sample_rate
            self.mu = config.mu
        else:
            self.sample_rate = 16000
            self.mu = 255

        # 音響モデル
        self.acoustic_config = OmegaConf.load(model_dir / "acoustic_model.yaml")
        self.acoustic_model = instantiate(self.acoustic_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "acoustic_model.pth",
            map_location=device,
        )
        self.acoustic_model.load_state_dict(checkpoint["state_dict"])
        self.acoustic_out_scaler = StandardScaler(
            np.load(model_dir / "out_tacotron_scaler_mean.npy"),
            np.load(model_dir / "out_tacotron_scaler_var.npy"),
            np.load(model_dir / "out_tacotron_scaler_scale.npy"),
        )
        self.acoustic_model.eval()

        # WaveNet vocoder
        self.wavenet_config = OmegaConf.load(model_dir / "wavenet_model.yaml")
        self.wavenet_model = instantiate(self.wavenet_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "wavenet_model.pth",
            map_location=device,
        )
        self.wavenet_model.load_state_dict(checkpoint["state_dict"])
        self.wavenet_model.eval()
        self.wavenet_model.remove_weight_norm_()

    def __repr__(self):
        acoustic_str = json.dumps(
            OmegaConf.to_container(self.acoustic_config["netG"]),
            sort_keys=False,
            indent=4,
        )
        wavenet_str = json.dumps(
            OmegaConf.to_container(self.wavenet_config["netG"]),
            sort_keys=False,
            indent=4,
        )

        return f"""Tacotron2 TTS (sampling rate: {self.sample_rate})

Acoustic model: {acoustic_str}
Vocoder model: {wavenet_str}
"""

    def set_device(self, device):
        """Set device for the TTS models

        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.acoustic_model.to(device)
        self.wavenet_model.to(device)

    @torch.no_grad()
    def tts(self, text, griffin_lim=False, tqdm=tqdm):
        """Run TTS

        Args:
            text (str): Input text
            griffin_lim (bool, optional): Use Griffin-Lim algorithm or not. Defaults to False.
            tqdm (object, optional): tqdm object. Defaults to None.

        Returns:
            tuple: audio array (np.int16) and sampling rate (int)
        """
        # OpenJTalkを用いて言語特徴量の抽出
        contexts = pyopenjtalk.extract_fullcontext(text)
        # 韻律記号付き音素列に変換
        in_feats = text_to_sequence(pp_symbols(contexts))
        in_feats = torch.tensor(in_feats, dtype=torch.long).to(self.device)

        # (T, C)
        _, out_feats, _, _ = self.acoustic_model.inference(in_feats)

        if griffin_lim:
            # Griffin-Lim のアルゴリズムに基づく音声波形合成
            out_feats = out_feats.cpu().data.numpy()
            # 正規化の逆変換
            logmel = self.acoustic_out_scaler.inverse_transform(out_feats)
            gen_wav = logmelspectrogram_to_audio(logmel, self.sample_rate)
        else:
            # (B, T, C) -> (B, C, T)
            c = out_feats.view(1, -1, out_feats.size(-1)).transpose(1, 2)

            # 音声波形の長さを計算
            upsample_scale = np.prod(self.wavenet_model.upsample_scales)
            T = (
                c.shape[-1] - self.wavenet_model.aux_context_window * 2
            ) * upsample_scale

            # WaveNet ボコーダによる音声波形の生成
            # NOTE: 計算に時間を要するため、tqdm によるプログレスバーを利用します
            gen_wav = self.wavenet_model.inference(c, T, tqdm)

            # One-hot ベクトルから1次元の信号に変換
            gen_wav = gen_wav.max(1)[1].float().cpu().numpy().reshape(-1)

            # Mu-law 量子化の逆変換
            # NOTE: muは出力チャンネル数-1だと仮定
            gen_wav = inv_mulaw_quantize(gen_wav, self.wavenet_model.out_channels - 1)

        return self.post_process(gen_wav), self.sample_rate

    def post_process(self, wav):
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767.0).astype(np.int16)
        return wav


def randomize_tts_engine_(engine: Tacotron2TTS) -> Tacotron2TTS:
    # アテンションのパラメータの一部を強制的に乱数で初期化することで、学習済みモデルを破壊する
    torch.nn.init.normal_(engine.acoustic_model.decoder.attention.mlp_dec.weight.data)
    return engine
