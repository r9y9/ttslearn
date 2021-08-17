from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from ttslearn.train_util import ensure_divisible_by
from ttslearn.util import load_utt_list, optional_tqdm


@hydra.main(config_path="conf/generate_gta", config_name="config")
def my_app(config: DictConfig) -> None:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.acoustic.checkpoint),
        map_location=device,
    )
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_model.eval()

    rf = acoustic_model.decoder.reduction_factor

    in_dir = Path(to_absolute_path(config.in_dir))
    out_dir = Path(to_absolute_path(config.out_dir))
    gta_dir = Path(to_absolute_path(config.gta_dir))
    gta_dir.mkdir(parents=True, exist_ok=True)

    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    in_feats_files = [in_dir / f"{utt_id.strip()}-feats.npy" for utt_id in utt_ids]
    out_feats_files = [out_dir / f"{utt_id.strip()}-feats.npy" for utt_id in utt_ids]
    wave_files = [out_dir / f"{utt_id.strip()}-wave.npy" for utt_id in utt_ids]

    # Run synthesis for each utt.
    for utt_id, in_feats_file, out_feats_file, wave_file in optional_tqdm(
        config.tqdm, desc="Utterance"
    )(zip(utt_ids, in_feats_files, out_feats_files, wave_files)):
        wave = np.load(wave_file)
        out_feats = np.load(out_feats_file)
        assert len(wave) % len(out_feats) == 0
        hop_size = len(wave) // len(out_feats)

        # Adjust feature length for reduction factor > 1.
        org_len = len(out_feats)
        out_feats = ensure_divisible_by(out_feats, rf)
        diff_len = org_len - len(out_feats)
        if diff_len > 0:
            wave = wave[: -diff_len * hop_size]
        assert len(wave) % len(out_feats) == 0

        # Send data to device
        in_feats = torch.from_numpy(np.load(in_feats_file)).unsqueeze(0).to(device)
        in_lens = torch.tensor([in_feats.shape[-1]], dtype=torch.long, device=device)
        out_feats = torch.from_numpy(out_feats).unsqueeze(0).to(device)

        # Run teacher-forcing
        with torch.no_grad():
            _, pred_out_feats, _, _ = acoustic_model(in_feats, in_lens, out_feats)
        assert pred_out_feats.shape == out_feats.shape

        # Save GTA features and its aligned waveform
        pred_out_feats = pred_out_feats.squeeze(0).to("cpu").numpy()
        assert len(wave) % len(pred_out_feats) == 0
        np.save(
            gta_dir / f"{utt_id}-feats.npy",
            pred_out_feats.astype(np.float32),
            allow_pickle=False,
        )
        np.save(
            gta_dir / f"{utt_id}-wave.npy",
            wave.astype(np.float32),
            allow_pickle=False,
        )


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
