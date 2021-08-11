from functools import partial
from pathlib import Path

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from tqdm import tqdm
from ttslearn.dnntts.gen import predict_acoustic, predict_duration
from ttslearn.util import load_utt_list, optional_tqdm
from ttslearn.wavenet.gen import gen_waveform


def synthesis(
    device,
    labels,
    binary_dict,
    numeric_dict,
    duration_model,
    duration_config,
    duration_in_scaler,
    duration_out_scaler,
    logf0_model,
    logf0_config,
    logf0_in_scaler,
    logf0_out_scaler,
    wavenet_model,
    wavenet_in_scaler,
    tqdm=tqdm,
):
    # Predict durations
    if duration_model is not None:
        durations = predict_duration(
            device,
            labels,
            duration_model,
            duration_config,
            duration_in_scaler,
            duration_out_scaler,
            binary_dict,
            numeric_dict,
        )
        labels.set_durations(durations)

    # Predict acoustic features
    logf0 = predict_acoustic(
        device,
        labels,
        logf0_model,
        logf0_config,
        logf0_in_scaler,
        logf0_out_scaler,
        binary_dict,
        numeric_dict,
        mlpg=False,
    )

    # Waveform generation
    gen_wav = gen_waveform(
        device,
        labels,
        logf0,
        wavenet_model,
        wavenet_in_scaler,
        binary_dict,
        numeric_dict,
        tqdm,
    )

    return gen_wav


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # duration
    duration_config = OmegaConf.load(to_absolute_path(config.duration.model_yaml))
    duration_model = hydra.utils.instantiate(duration_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.duration.checkpoint),
        map_location=device,
    )
    duration_model.load_state_dict(checkpoint["state_dict"])
    duration_in_scaler = joblib.load(to_absolute_path(config.duration.in_scaler_path))
    duration_out_scaler = joblib.load(to_absolute_path(config.duration.out_scaler_path))
    duration_model.eval()

    # logf0 prediction model
    logf0_config = OmegaConf.load(to_absolute_path(config.logf0.model_yaml))
    logf0_model = hydra.utils.instantiate(logf0_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.logf0.checkpoint),
        map_location=device,
    )
    logf0_model.load_state_dict(checkpoint["state_dict"])
    logf0_in_scaler = joblib.load(to_absolute_path(config.logf0.in_scaler_path))
    logf0_out_scaler = joblib.load(to_absolute_path(config.logf0.out_scaler_path))
    logf0_model.eval()

    # WaveNet
    wavenet_config = OmegaConf.load(to_absolute_path(config.wavenet.model_yaml))
    wavenet_model = hydra.utils.instantiate(wavenet_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.wavenet.checkpoint),
        map_location=device,
    )
    wavenet_model.load_state_dict(checkpoint["state_dict"])
    wavenet_in_scaler = joblib.load(to_absolute_path(config.wavenet.in_scaler_path))
    wavenet_model.eval()
    wavenet_model.remove_weight_norm_()

    binary_dict, numeric_dict = hts.load_question_set(to_absolute_path(config.qst_path))

    in_dir = Path(to_absolute_path(config.in_dir))
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    if config.reverse:
        utt_ids = utt_ids[::-1]
    lab_files = [in_dir / f"{utt_id.strip()}.lab" for utt_id in utt_ids]
    if config.num_eval_utts is not None and config.num_eval_utts > 0:
        lab_files = lab_files[: config.num_eval_utts]

    if config.tqdm == "tqdm":
        _tqdm = partial(tqdm, desc="wavenet generation", leave=False)
    else:
        _tqdm = None
    for lab_file in optional_tqdm(config.tqdm, desc="Utterance")(lab_files):
        labels = hts.load(lab_file).round_()

        wav = synthesis(
            device,
            labels,
            binary_dict,
            numeric_dict,
            duration_model,
            duration_config,
            duration_in_scaler,
            duration_out_scaler,
            logf0_model,
            logf0_config,
            logf0_in_scaler,
            logf0_out_scaler,
            wavenet_model,
            wavenet_in_scaler,
            _tqdm,
        )
        wav = np.clip(wav, -1.0, 1.0)

        utt_id = lab_file.stem
        out_wav_path = out_dir / f"{utt_id}.wav"
        wavfile.write(
            out_wav_path,
            rate=config.sample_rate,
            data=(wav * 32767.0).astype(np.int16),
        )


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
