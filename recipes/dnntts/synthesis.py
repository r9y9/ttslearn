from pathlib import Path

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from ttslearn.dnntts.gen import gen_waveform, predict_acoustic, predict_duration
from ttslearn.util import load_utt_list, optional_tqdm


def synthesis(
    device,
    sample_rate,
    labels,
    binary_dict,
    numeric_dict,
    duration_model,
    duration_config,
    duration_in_scaler,
    duration_out_scaler,
    acoustic_model,
    acoustic_config,
    acoustic_in_scaler,
    acoustic_out_scaler,
    post_filter=False,
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
    acoustic_features = predict_acoustic(
        device,
        labels,
        acoustic_model,
        acoustic_config,
        acoustic_in_scaler,
        acoustic_out_scaler,
        binary_dict,
        numeric_dict,
    )

    # Waveform generation
    gen_wav = gen_waveform(
        sample_rate,
        acoustic_features,
        acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        acoustic_config.num_windows,
        post_filter,
    )

    return gen_wav


def run_synthesis(out_dir, device, sample_rate, lab_file, *args, **kwargs):
    labels = hts.load(lab_file).round_()
    wav = synthesis(device, sample_rate, labels, *args, **kwargs)
    wav = np.clip(wav, -1.0, 1.0)

    utt_id = lab_file.stem
    out_wav_path = out_dir / f"{utt_id}.wav"
    wavfile.write(
        out_wav_path,
        rate=sample_rate,
        data=(wav * 32767.0).astype(np.int16),
    )
    return wav


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    binary_dict, numeric_dict = hts.load_question_set(to_absolute_path(config.qst_path))

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

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.acoustic.checkpoint),
        map_location=device,
    )
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_in_scaler = joblib.load(to_absolute_path(config.acoustic.in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
    acoustic_model.eval()

    in_dir = Path(to_absolute_path(config.in_dir))
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    if config.reverse:
        utt_ids = utt_ids[::-1]
    lab_files = [in_dir / f"{utt_id.strip()}.lab" for utt_id in utt_ids]
    if config.num_eval_utts is not None and config.num_eval_utts > 0:
        lab_files = lab_files[: config.num_eval_utts]

    # Run synthesis for each utt.
    for lab_file in optional_tqdm(config.tqdm, desc="Utterance")(lab_files):
        labels = hts.load(lab_file).round_()

        wav = synthesis(
            device,
            config.sample_rate,
            labels,
            binary_dict,
            numeric_dict,
            duration_model,
            duration_config,
            duration_in_scaler,
            duration_out_scaler,
            acoustic_model,
            acoustic_config,
            acoustic_in_scaler,
            acoustic_out_scaler,
            config.post_filter,
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
