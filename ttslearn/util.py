# Acknowledgements:
# mask-related functions were adapted from https://github.com/espnet/espnet

import importlib
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pkg_resources
import torch

# see COPYING for the license of the audio file.
EXAMPLE_AUDIO = "_example_data/BASIC5000_0001.wav"
EXAMPLE_LABEL = "_example_data/BASIC5000_0001.lab"
EXAMPLE_MONO_LABEL = "_example_data/BASIC5000_0001_mono.lab"
EXAMPLE_QST = "_example_data/qst1.hed"


def init_seed(seed):
    """Initialize random seed.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dynamic_import(name: str) -> Any:
    """Dynamic import

    Args:
        name (str): module_name + ":" + class_name

    Returns:
        Any: class object
    """
    mod_name, class_name = name.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, class_name)


def make_pad_mask(lengths, maxlen=None):
    """Make mask for padding frames

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def make_non_pad_mask(lengths, maxlen=None):
    """Make mask for non-padding frames

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    return ~make_pad_mask(lengths, maxlen)


def example_audio_file() -> str:
    """Get the path to an included audio example file.

    Examples
    --------
    >>> from scipy.io import wavfile
    >>> fs, x = wavfile.read(pysptk.util.example_audio_file())

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, label="cmu_us_awb_arctic arctic_a0007.wav")
    >>> plt.xlim(0, len(x))
    >>> plt.legend()

    """

    return pkg_resources.resource_filename(__name__, EXAMPLE_AUDIO)


def example_label_file(mono=False):
    """Get the path to an included label file.

    Args:
        mono (bool, optional): If True, return monophonic label file.
            Default: False

    Returns:
        str: path to an example label file
    """
    if mono:
        return pkg_resources.resource_filename(__name__, EXAMPLE_MONO_LABEL)
    return pkg_resources.resource_filename(__name__, EXAMPLE_LABEL)


def example_qst_file():
    """Get the path to an included question set file.

    Returns:
        str: path to an example question file.
    """
    return pkg_resources.resource_filename(__name__, EXAMPLE_QST)


def pad_1d(x, max_len, constant_values=0):
    """Pad a 1d-tensor.

    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        (0, max_len - len(x)),
        mode="constant",
        constant_values=constant_values,
    )
    return x


def pad_2d(x, max_len, constant_values=0):
    """Pad a 2d-tensor.

    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x


def load_utt_list(utt_list):
    """Load a list of utterances.

    Args:
        utt_list (str): path to a file containing a list of utterances

    Returns:
        List[str]: list of utterances
    """
    utt_ids = []
    with open(utt_list) as f:
        for utt_id in f:
            utt_id = utt_id.strip()
            if len(utt_id) > 0:
                utt_ids.append(utt_id)
    return utt_ids


def trim_silence(feats, labels, start_sec=0.05, end_sec=0.1, shift_sec=0.005):
    """Trim silence from input features.

    Args:
        feats (np.ndarray): input features
        labels (np.ndarray): labels
        start_sec (float, optional): start time of the trim
        end_sec (float, optional): end time of the trim
        shift_sec (float, optional): shift of the trim

    Returns:
        np.ndarray: trimmed features
    """
    assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
    start_frame = int(labels.start_times[1] / 50000)
    end_frame = int(labels.end_times[-2] / 50000)
    start_frame = max(0, start_frame - int(start_sec / shift_sec))
    end_frame = min(len(feats), end_frame + int(end_sec / shift_sec))
    feats = feats[start_frame:end_frame]
    return feats


def find_feats(directory, utt_id, typ="out_duration", ext="-feats.npy"):
    """Find features for a given utterance.

    Args:
        directory (str): directory to search
        utt_id (str): utterance id
        typ (str, optional): type of the feature. Default: "out_duration"
        ext (str, optional): extension of the feature. Default: "-feats.npy"

    Returns:
        str: path to the feature file
    """
    if isinstance(directory, str):
        directory = Path(directory)
    ps = sorted(directory.rglob(f"**/{typ}/{utt_id}{ext}"))
    return ps[0]


def find_lab(directory, utt_id):
    """Find label for a given utterance.

    Args:
        directory (str): directory to search
        utt_id (str): utterance id

    Returns:
        str: path to the label file
    """
    if isinstance(directory, str):
        directory = Path(directory)
    ps = sorted(directory.rglob(f"{utt_id}.lab"))
    assert len(ps) == 1
    return ps[0]


def lab2phonemes(labels):
    """Convert labels to phonemes.

    Args:
        labels (str): path to a label file

    Returns:
        List[str]: phoneme sequence
    """
    phonemes = []
    for c in labels.contexts:
        if "-" in c:
            ph = c.split("-")[1].split("+")[0]
        else:
            ph = c
        phonemes.append(ph)
    return phonemes


def optional_tqdm(tqdm_mode, **kwargs):
    """Get a tqdm object.

    Args:
        tqdm_mode (str): tqdm mode
        **kwargs: keyword arguments for tqdm

    Returns:
        callable: tqdm object or an identity function
    """
    if tqdm_mode == "tqdm":
        from tqdm import tqdm

        return partial(tqdm, **kwargs)
    elif tqdm_mode == "tqdm-notebook":
        from tqdm.notebook import tqdm

        return partial(tqdm, **kwargs)

    return lambda x: x


class StandardScaler:
    """sklearn.preprocess.StandardScaler like class with only
    transform functionality

    Args:
        mean (np.ndarray): mean
        std (np.ndarray): standard deviation
    """

    def __init__(self, mean, var, scale):
        self.mean_ = mean
        self.var_ = var
        # NOTE: scale may not exactly same as np.sqrt(var)
        self.scale_ = scale

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_
