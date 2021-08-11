import json
from pathlib import Path

import numpy as np
import pyopenjtalk
import torch
from hydra.utils import instantiate
from nnmnkwii.io import hts
from omegaconf import OmegaConf
from tqdm import tqdm
from ttslearn.dnntts.gen import predict_acoustic, predict_duration
from ttslearn.pretrained import retrieve_pretrained_model
from ttslearn.util import StandardScaler
from ttslearn.wavenet.gen import gen_waveform


class WaveNetTTS(object):
    """WaveNet-based text-to-speech

    Args:
        model_dir (str): model directory. A pre-trained model (ID: ``wavenettts``)
            is used if None.
        device (str): cpu or cuda.

    Examples:
        >>> from ttslearn.dnntts import WaveNetTTS
        >>> engine = WaveNetTTS()
        >>> wav, sr = engine.tts("今日もいい天気ですね。")
    """

    def __init__(self, model_dir=None, device="cpu"):
        self.device = device

        if model_dir is None:
            model_dir = retrieve_pretrained_model("wavenettts")
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

        # 対数基本周波数の予測モデル
        self.logf0_config = OmegaConf.load(model_dir / "logf0_model.yaml")
        self.logf0_model = instantiate(self.logf0_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "logf0_model.pth",
            map_location=device,
        )
        self.logf0_model.load_state_dict(checkpoint["state_dict"])
        self.logf0_in_scaler = StandardScaler(
            np.load(model_dir / "in_logf0_scaler_mean.npy"),
            np.load(model_dir / "in_logf0_scaler_var.npy"),
            np.load(model_dir / "in_logf0_scaler_scale.npy"),
        )
        self.logf0_out_scaler = StandardScaler(
            np.load(model_dir / "out_logf0_scaler_mean.npy"),
            np.load(model_dir / "out_logf0_scaler_var.npy"),
            np.load(model_dir / "out_logf0_scaler_scale.npy"),
        )
        self.logf0_model.eval()

        # WaveNet
        self.wavenet_config = OmegaConf.load(model_dir / "wavenet_model.yaml")
        self.wavenet_model = instantiate(self.wavenet_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "wavenet_model.pth",
            map_location=device,
        )
        self.wavenet_model.load_state_dict(checkpoint["state_dict"])
        self.wavenet_in_scaler = StandardScaler(
            np.load(model_dir / "in_wavenet_scaler_mean.npy"),
            np.load(model_dir / "in_wavenet_scaler_var.npy"),
            np.load(model_dir / "in_wavenet_scaler_scale.npy"),
        )
        self.wavenet_model.eval()
        self.wavenet_model.remove_weight_norm_()

    def __repr__(self):
        duration_str = json.dumps(
            OmegaConf.to_container(self.duration_config.netG),
            sort_keys=False,
            indent=4,
        )
        logf0_str = json.dumps(
            OmegaConf.to_container(self.logf0_config.netG), sort_keys=False, indent=4
        )
        wavenet_str = json.dumps(
            OmegaConf.to_container(self.wavenet_config.netG),
            sort_keys=False,
            indent=4,
        )

        return f"""WaveNet TTS (sampling rate: {self.sample_rate})

Duration model: {duration_str}
Log-f0 model: {logf0_str}
WaveNet: {wavenet_str}
"""

    def set_device(self, device):
        """Set device for the TTS models

        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.duration_model.to(device)
        self.logf0_model.to(device)
        self.wavenet_model.to(device)

    @torch.no_grad()
    def tts(self, text, tqdm=tqdm):
        """Run TTS

        Args:
            text (str): Input text
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

        # 対数基本周波数（および有声無声フラグ）の予測
        logf0_vuv = predict_acoustic(
            self.device,
            labels,
            self.logf0_model,
            self.logf0_config,
            self.logf0_in_scaler,
            self.logf0_out_scaler,
            self.binary_dict,
            self.numeric_dict,
            mlpg=False,
        )

        # WaveNetによる音声波形の生成
        wav = gen_waveform(
            self.device,
            labels,
            logf0_vuv,
            self.wavenet_model,
            self.wavenet_in_scaler,
            self.binary_dict,
            self.numeric_dict,
            tqdm,
        )

        return self.post_process(wav), self.sample_rate

    def post_process(self, wav):
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767.0).astype(np.int16)
        return wav
