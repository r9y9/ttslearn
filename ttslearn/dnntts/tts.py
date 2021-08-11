import json
from pathlib import Path

import numpy as np
import pyopenjtalk
import torch
from hydra.utils import instantiate
from nnmnkwii.io import hts
from omegaconf import OmegaConf
from ttslearn.dnntts.gen import gen_waveform, predict_acoustic, predict_duration
from ttslearn.pretrained import retrieve_pretrained_model
from ttslearn.util import StandardScaler


class DNNTTS(object):
    """DNN-based text-to-speech

    Args:
        model_dir (str): model directory. A pre-trained model (ID: ``dnntts``)
            is used if None.
        device (str): cpu or cuda.

    Examples:

    .. plot::

        from ttslearn.dnntts import DNNTTS
        import matplotlib.pyplot as plt

        engine = DNNTTS()
        wav, sr = engine.tts("日本語音声合成のデモです。")

        fig, ax = plt.subplots(figsize=(8,2))
        librosa.display.waveplot(wav.astype(np.float32), sr, ax=ax)
    """

    def __init__(self, model_dir=None, device="cpu"):
        self.device = device

        if model_dir is None:
            model_dir = retrieve_pretrained_model("dnntts")
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # search for config.yaml
        if (model_dir / "config.yaml").exists():
            config = OmegaConf.load(model_dir / "config.yaml")
            self.sample_rate = config.sample_rate
        else:
            self.sample_rate = 16000

        # qst
        self.binary_dict, self.numeric_dict = hts.load_question_set(
            model_dir / "qst.hed"
        )

        # 継続長モデル
        self.duration_config = OmegaConf.load(model_dir / "duration_model.yaml")
        self.duration_model = instantiate(self.duration_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "duration_model.pth",
            map_location=device,
        )
        self.duration_model.load_state_dict(checkpoint["state_dict"])

        self.duration_in_scaler = StandardScaler(
            np.load(model_dir / "in_duration_scaler_mean.npy"),
            np.load(model_dir / "in_duration_scaler_var.npy"),
            np.load(model_dir / "in_duration_scaler_scale.npy"),
        )
        self.duration_out_scaler = StandardScaler(
            np.load(model_dir / "out_duration_scaler_mean.npy"),
            np.load(model_dir / "out_duration_scaler_var.npy"),
            np.load(model_dir / "out_duration_scaler_scale.npy"),
        )
        self.duration_model.eval()

        # 音響モデル
        self.acoustic_config = OmegaConf.load(model_dir / "acoustic_model.yaml")
        self.acoustic_model = instantiate(self.acoustic_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "acoustic_model.pth",
            map_location=device,
        )
        self.acoustic_model.load_state_dict(checkpoint["state_dict"])
        self.acoustic_in_scaler = StandardScaler(
            np.load(model_dir / "in_acoustic_scaler_mean.npy"),
            np.load(model_dir / "in_acoustic_scaler_var.npy"),
            np.load(model_dir / "in_acoustic_scaler_scale.npy"),
        )
        self.acoustic_out_scaler = StandardScaler(
            np.load(model_dir / "out_acoustic_scaler_mean.npy"),
            np.load(model_dir / "out_acoustic_scaler_var.npy"),
            np.load(model_dir / "out_acoustic_scaler_scale.npy"),
        )
        self.acoustic_model.eval()

    def __repr__(self):
        duration_str = json.dumps(
            OmegaConf.to_container(self.duration_config.netG),
            sort_keys=False,
            indent=4,
        )
        acoustic_str = json.dumps(
            OmegaConf.to_container(self.acoustic_config.netG),
            sort_keys=False,
            indent=4,
        )

        return f"""DNNTTS (sampling rate: {self.sample_rate})

Duration model: {duration_str}
Acoustic model: {acoustic_str}
"""

    def set_device(self, device):
        """Set device for the TTS models

        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.duration_model.to(device)
        self.acoustic_model.to(device)

    @torch.no_grad()
    def tts(self, text, post_filter=True, tqdm=None):
        """Run TTS

        Args:
            text (str): Input text
            post_filter (bool, optional): Use post-filter or not. Defaults to True.
            tqdm (object, optional): tqdm object. Defaults to None.

        Returns:
            tuple: audio array (np.int16) and sampling rate (int)
        """
        # OpenJTalkを用いて言語特徴量の抽出
        contexts = pyopenjtalk.extract_fullcontext(text)
        # HTS 形式に変換
        if hasattr(hts.HTSLabelFile, "create_from_contexts"):
            labels = hts.HTSLabelFile.create_from_contexts(contexts)
        else:
            labels = hts.load(None, contexts)

        # 音素継続長の予測
        durations = predict_duration(
            self.device,
            labels,
            self.duration_model,
            self.duration_config,
            self.duration_in_scaler,
            self.duration_out_scaler,
            self.binary_dict,
            self.numeric_dict,
        )
        labels.set_durations(durations)

        # 音響特徴量の予測
        acoustic_features = predict_acoustic(
            self.device,
            labels,
            self.acoustic_model,
            self.acoustic_config,
            self.acoustic_in_scaler,
            self.acoustic_out_scaler,
            self.binary_dict,
            self.numeric_dict,
        )

        # ボコーダを用いて音声波形の生成
        wav = gen_waveform(
            self.sample_rate,
            acoustic_features,
            self.acoustic_config.stream_sizes,
            self.acoustic_config.has_dynamic_features,
            self.acoustic_config.num_windows,
            post_filter=post_filter,
        )

        return self.post_process(wav), self.sample_rate

    def post_process(self, wav):
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767.0).astype(np.int16)
        return wav
