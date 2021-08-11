from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from tqdm.auto import tqdm
from ttslearn.pretrained import (
    create_tts_engine,
    get_available_model_ids,
    is_pretrained_model_ready,
    retrieve_pretrained_model,
)

OUT_DIR = Path(__file__).parent / "out_dir"
OUT_DIR.mkdir(exist_ok=True)


def test_is_pretrained_model_ready():
    # warmup
    create_tts_engine("dnntts").tts("test")
    # should exist
    assert is_pretrained_model_ready("dnntts")
    # I wish...
    assert not is_pretrained_model_ready("super_sugoi_tsuyoi_model")


def test_retrieve_pretrained_model():
    # warmup
    create_tts_engine("dnntts").tts("test")

    # shouldn't raise
    retrieve_pretrained_model("dnntts")

    with pytest.raises(ValueError):
        retrieve_pretrained_model("super_sugoi_tsuyoi_model")


# Test if the results sound okay. Check the generated wav files after running the test
def test_all_pretraind_models():
    for idx, name in enumerate(get_available_model_ids()):
        if not is_pretrained_model_ready(name):
            print(f"Pretrained model does not exist: {name}")
            continue
        print(idx, name)
        engine = create_tts_engine(name)
        if hasattr(engine, "spks") and engine.spks is not None:
            assert engine.spk2id is not None
            wav, sr = engine.tts("ありがとうございました", tqdm=tqdm, spk_id=1)
        else:
            wav, sr = engine.tts("ありがとうございました", tqdm=tqdm)

        assert wav.dtype == np.int16
        wav = (wav / np.abs(wav).max() * 32767.0).astype(np.int16)
        wavfile.write(OUT_DIR / f"{idx:02d}_test_{name}.wav", sr, wav)

        assert len(wav) > 0
