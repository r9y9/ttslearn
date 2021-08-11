import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyopenjtalk
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from parallel_wavegan.utils import load_model
from ttslearn.pretrained import retrieve_pretrained_model
from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence


class Tacotron2PWGTTS(object):
    """Fast Tacotron 2 based text-to-speech with Parallel WaveGAN

    The WaveNet vocoder in Tacotron 2 is replaced with Parallel WaveGAN for
    fast real-time inference.
    Both single-speaker and multi-speaker Tacotron are supported.

    Args:
        model_dir (str): model directory. A pre-trained model (ID: ``tacotron2_pwg_jsut24k``)
            is used if None.
        device (str): cpu or cuda.

    Examples:

        Singler-speaker TTS

        >>> from ttslearn.contrib import Tacotron2PWGTTS
        >>> engine = Tacotron2PWGTTS()
        >>> wav, sr = engine.tts("発展的な音声合成です！")

        Multi-speaker TTS

        >>> from ttslearn.contrib import Tacotron2PWGTTS
        >>> from ttslearn.pretrained import retrieve_pretrained_model
        >>> model_dir = retrieve_pretrained_model("multspk_tacotron2_pwg_jvs24k")
        >>> engine = Tacotron2PWGTTS(model_dir)
        >>> wav, sr = engine.tts("じぇーぶいえすコーパス10番目の話者です。", spk_id=10)

    .. note::

        This class supports not only `Parallel WaveGAN`_ but also any models supported in
        `kan-bayashi/ParallelWaveGAN`_.
        For example, HifiGAN or MelGAN can also be used without any change.

        .. _Parallel WaveGAN: https://arxiv.org/abs/1910.11480
        .. _kan-bayashi/ParallelWaveGAN: https://github.com/kan-bayashi/ParallelWaveGAN
    """

    def __init__(self, model_dir=None, device="cpu"):
        self.device = device

        if model_dir is None:
            model_dir = retrieve_pretrained_model("tacotron2_pwg_jsut16k")
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # search for config.yaml
        if (model_dir / "config.yaml").exists():
            config = OmegaConf.load(model_dir / "config.yaml")
            self.sample_rate = config.sample_rate
        else:
            self.sample_rate = 16000

        # 話者情報 (for multi-speaker models)
        if (model_dir / "spks").exists():
            # 話者名を取得
            with open(model_dir / "spks", "r") as f:
                self.spks: Optional[List[str]] = [line.strip() for line in f]
            # 話者名と話者ID（数値）の辞書の構築
            with open(model_dir / "spk2id") as f:
                self.spk2id: Optional[Dict[str, int]] = {}
                for line in f:
                    k, v = line.strip().split(":")
                    self.spk2id[k] = int(v)
        else:
            self.spks, self.spk2id = None, None

        # 音響モデル
        self.acoustic_config = OmegaConf.load(model_dir / "acoustic_model.yaml")
        self.acoustic_model = instantiate(self.acoustic_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "acoustic_model.pth",
            map_location=device,
        )
        self.acoustic_model.load_state_dict(checkpoint["state_dict"])
        self.acoustic_model.eval()

        # parallel_wavegan
        self.vocoder_config = OmegaConf.load(model_dir / "vocoder_model.yaml")
        self.vocoder = load_model(
            model_dir / "vocoder_model.pth", config=self.vocoder_config
        ).to(device)
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

    def __repr__(self):
        acoustic_str = json.dumps(
            OmegaConf.to_container(self.acoustic_config.netG),
            sort_keys=False,
            indent=4,
        )
        vocoder_params = {
            "generator_type": self.vocoder_config.get(
                "generator_type", "ParallelWaveGANGenerator"  # type: ignore
            ),
            "generator_params": OmegaConf.to_container(
                self.vocoder_config.generator_params
            ),
        }

        vocoder_str = json.dumps(
            vocoder_params,
            sort_keys=False,
            indent=4,
        )

        return f"""Tacotron2 TTS (sampling rate: {self.sample_rate})

Acoustic model: {acoustic_str}
Vocoder model: {vocoder_str}
"""

    def set_device(self, device):
        """Set device for the TTS models

        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.acoustic_model.to(device)
        self.vocoder.to(device)

    @torch.no_grad()
    def tts(self, text, tqdm=None, spk_id=None):
        """Run TTS

        Args:
            text (str): Input text
            tqdm (obj): tqdm progress bar
            spk_id (int): speaker id. This should be only specified for multi-speaker models.

        Returns:
            tuple: audio array (np.int16) and sampling rate (int)
        """
        # OpenJTalkを用いて言語特徴量の抽出
        labels = pyopenjtalk.extract_fullcontext(text)
        # 音素 + 韻律記号に変換
        in_feats = text_to_sequence(pp_symbols(labels))
        in_feats = torch.tensor(in_feats, dtype=torch.long).to(self.device)

        if hasattr(self.acoustic_model, "spk_embed"):
            assert spk_id is not None
            spk_id = torch.tensor([spk_id], dtype=torch.long).to(self.device)
            _, out_feats, _, _ = self.acoustic_model.inference(in_feats, spk_id)
        else:
            # (T, C)
            _, out_feats, _, _ = self.acoustic_model.inference(in_feats)

        # parallel_wavegan による音声波形の生成
        gen_wav = self.vocoder.inference(out_feats).view(-1).to("cpu").numpy()

        return self.post_process(gen_wav), self.sample_rate

    def post_process(self, wav):
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767.0).astype(np.int16)
        return wav
