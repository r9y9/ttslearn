import os
import shutil
import tarfile
from os.path import join
from pathlib import Path
from urllib.request import urlretrieve

from tqdm.auto import tqdm
from ttslearn.util import dynamic_import

_urls = {"v0.2.0": "https://github.com/r9y9/ttslearn/releases/download/v0.2.0"}

DEFAULT_CACHE_DIR = join(os.environ["HOME"], ".cache", "ttslearn")
CACHE_DIR = os.environ.get("TTSLEARN_CACHE_DIR", DEFAULT_CACHE_DIR)


model_registry = {
    "dnntts": {
        "url": f"{_urls['v0.2.0']}/dnntts.tar.gz",
        "_target_": "ttslearn.dnntts:DNNTTS",
    },
    "wavenettts": {
        "url": f"{_urls['v0.2.0']}/wavenettts.tar.gz",
        "_target_": "ttslearn.wavenet:WaveNetTTS",
    },
    "tacotron2": {
        "url": f"{_urls['v0.2.0']}/tacotron2.tar.gz",
        "_target_": "ttslearn.tacotron:Tacotron2TTS",
    },
    "tacotron2_pwg_jsut16k": {
        "url": f"{_urls['v0.2.0']}/tacotron2_pwg_jsut16k.tar.gz",
        "_target_": "ttslearn.contrib:Tacotron2PWGTTS",
    },
    "tacotron2_pwg_jsut24k": {
        "url": f"{_urls['v0.2.0']}/tacotron2_pwg_jsut24k.tar.gz",
        "_target_": "ttslearn.contrib:Tacotron2PWGTTS",
    },
    "multspk_tacotron2_pwg_jvs16k": {
        "url": f"{_urls['v0.2.0']}/multspk_tacotron2_pwg_jvs16k.tar.gz",
        "_target_": "ttslearn.contrib:Tacotron2PWGTTS",
    },
    "multspk_tacotron2_pwg_jvs24k": {
        "url": f"{_urls['v0.2.0']}/multspk_tacotron2_pwg_jvs24k.tar.gz",
        "_target_": "ttslearn.contrib:Tacotron2PWGTTS",
    },
}


def create_tts_engine(name, *args, **kwargs):
    """Create TTS engine from official pretrained models.

    Args:
        name (str): Pre-trained model name
        args (list): Additional args for instanciation
        kwargs (dict): Additional kwargs for instanciation

    Returns:
        object: instance of TTS engine

    Examples:
        >>> from ttslearn.pretrained import create_tts_engine
        >>> create_tts_engine("dnntts")
        DNNTTS (sampling rate: 16000)
    """
    if name not in model_registry:
        s = ""
        for model_id in get_available_model_ids():
            s += f"'{model_id}'\n"
        raise ValueError(
            f"""
Pretrained model '{name}' does not exist!

Available models:
{s[:-1]}"""
        )

    # download if not exists
    model_dir = retrieve_pretrained_model(name)

    # create an instance
    return dynamic_import(model_registry[name]["_target_"])(model_dir, *args, **kwargs)


def get_available_model_ids():
    """Get available pretrained model names.

    Returns:
        list: List of available pretrained model names.

    Examples:
        >>> from ttslearn.pretrained import get_available_model_ids
        >>> get_available_model_ids()[:3]
        ['dnntts', 'wavenettts', 'tacotron2']

    """
    return list(model_registry.keys())


# https://github.com/tqdm/tqdm#hooks-and-callbacks
class _TqdmUpTo(tqdm):  # type: ignore
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)


def is_pretrained_model_ready(name):
    out_dir = Path(CACHE_DIR) / name
    if out_dir.exists() and len(list(out_dir.glob("*.pth"))) == 0:
        return False
    return out_dir.exists()


def retrieve_pretrained_model(name):
    """Retrieve pretrained model from local cache or download from GitHub.

    Args:
        name (str): Name of pretrained model.

    Returns:
        str: Path to the pretrained model.

    Raises:
        ValueError: If the pretrained model is not found.

    Examples:
        >>> from ttslearn.pretrained import retrieve_pretrained_model
        >>> from ttslearn.contrib import Tacotron2PWGTTS
        >>> model_dir = retrieve_pretrained_model("tacotron2_pwg_jsut24k")
        >>> engine = Tacotron2PWGTTS(model_dir=model_dir, device="cpu")
        >>> wav, sr = engine.tts("センパイ、かっこいいです、ほれちゃいます！")
    """
    global model_registry
    if name not in model_registry:
        s = ""
        for model_id in get_available_model_ids():
            s += f"'{model_id}'\n"
        raise ValueError(
            f"""
Pretrained model '{name}' does not exist!

Available models:
{s[:-1]}"""
        )

    url = model_registry[name]["url"]
    # NOTE: assuming that filename and extracted is the same
    out_dir = Path(CACHE_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(CACHE_DIR) / f"{name}.tar.gz"

    # re-download models
    if out_dir.exists() and len(list(out_dir.glob("*.pth"))) == 0:
        shutil.rmtree(out_dir)

    if not out_dir.exists():
        print('Downloading: "{}"'.format(url))
        with _TqdmUpTo(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=f"{name}.tar.gz",
        ) as t:  # all optional kwargs
            urlretrieve(url, filename, reporthook=t.update_to)
            t.total = t.n
        with tarfile.open(filename, mode="r|gz") as f:
            f.extractall(path=CACHE_DIR)
        os.remove(filename)

    return out_dir
