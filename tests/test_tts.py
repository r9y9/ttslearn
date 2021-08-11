import numpy as np
import pytest
from ttslearn.dnntts import DNNTTS
from ttslearn.pretrained import create_tts_engine, is_pretrained_model_ready
from ttslearn.tacotron import Tacotron2TTS
from ttslearn.wavenet import WaveNetTTS


def test_dnntts():
    engine = DNNTTS()
    wav, sr = engine.tts("こんにちは")
    assert wav.dtype == np.int16
    assert sr == 16000


@pytest.mark.skipif(
    not is_pretrained_model_ready("wavenettts"), reason="wavenettts model not ready"
)
def test_wavenet():
    engine = WaveNetTTS()
    wav, sr = engine.tts("あ")
    assert wav.dtype == np.int16
    assert sr == 16000


@pytest.mark.skipif(
    not is_pretrained_model_ready("tacotron2"), reason="tacotron2 model not ready"
)
def test_tacotron():
    engine = Tacotron2TTS()
    wav, sr = engine.tts("あ", griffin_lim=True)
    assert wav.dtype == np.int16
    assert sr == 16000


def test_create_tts_engine():
    engine = create_tts_engine("dnntts")
    wav, sr = engine.tts("こんにちは")
    assert wav.dtype == np.int16
    assert sr == 16000
