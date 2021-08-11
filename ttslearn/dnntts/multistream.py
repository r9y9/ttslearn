import numpy as np
from nnmnkwii import paramgen


def get_windows(num_window):
    """Get windows for parameter generation

    Args:
        num_window (int): Number of windows (1, 2 or 3)

    Raises:
        ValueError: if not supported num_windows is specified

    Returns:
        list: list of windows
    """
    windows = [(0, 0, np.array([1.0]))]
    if num_window >= 2:
        windows.append((1, 1, np.array([-0.5, 0.0, 0.5])))
    if num_window >= 3:
        windows.append((1, 1, np.array([1.0, -2.0, 1.0])))

    if num_window >= 4:
        raise ValueError(f"Not supported num windows: {num_window}")

    return windows


def split_streams(inputs, stream_sizes):
    """Split streams from multi-stream features

    Args:
        inputs (array like): input 3-d array
        stream_sizes (list): sizes for each stream

    Returns:
        list: list of stream features
    """
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size in zip(start_indices, stream_sizes):
        if len(inputs.shape) == 3:
            s = inputs[:, :, start_idx : start_idx + size]
        else:
            s = inputs[:, start_idx : start_idx + size]
        ret.append(s)

    return ret


def get_static_stream_sizes(stream_sizes, has_dynamic_features, num_windows):
    """Get static sizes for each feature stream

    Args:
        stream_sizes (list): stream sizes
        has_dynamic_features (list): binary flags that indicates if steams have dynamic features
        num_windows (int): number of windows

    Returns:
        list: stream sizes
    """
    static_stream_sizes = np.array(stream_sizes)
    static_stream_sizes[has_dynamic_features] = (
        static_stream_sizes[has_dynamic_features] / num_windows
    )

    return static_stream_sizes


def get_static_features(
    inputs,
    num_windows,
    stream_sizes,
    has_dynamic_features,
    streams=None,
):
    """Get static features from static+dynamic features


    Args:
        inputs (array like): input 3-d or 2-d array
        num_windows (int): number of windows
        stream_sizes (list): stream sizes
        has_dynamic_features (list): binary flags that indicates if steams have dynamic features
        streams (list, optional): Streams of interests. Returns all streams if streams is None.
            Defaults to None.

    Returns:
        list: list of static features
    """
    _, D = inputs.shape
    if stream_sizes is None or (len(stream_sizes) == 1 and has_dynamic_features[0]):
        return inputs[:, : D // num_windows]
    if len(stream_sizes) == 1 and not has_dynamic_features[0]:
        return inputs
    if streams is None:
        streams = [True] * len(stream_sizes)

    # Multi stream case
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, v, enabled in zip(
        start_indices, stream_sizes, has_dynamic_features, streams
    ):
        if not enabled:
            continue
        if v:
            static_features = inputs[:, start_idx : start_idx + size // num_windows]
        else:
            static_features = inputs[:, start_idx : start_idx + size]
        ret.append(static_features)
    return ret


def multi_stream_mlpg(
    inputs,
    variances,
    windows,
    stream_sizes,
    has_dynamic_features,
    streams=None,
):
    """Split streams and do apply MLPG if stream has dynamic features

    Args:
        inputs (array like): input 3-d or 2-d array
        variances (array like): variances of input features
        windows (list): windows for parameter generation
        stream_sizes (list): stream sizes
        has_dynamic_features (list): binary flags that indicates if steams have dynamic features
        streams (list, optional): Streams of interests. Returns all streams if streams is None.
            Defaults to None.

    Raises:
        RuntimeError: if stream sizes are wrong

    Returns:
        array like: generated static features
    """
    T, D = inputs.shape
    if D != sum(stream_sizes):
        raise RuntimeError("You probably have specified wrong dimention params.")
    if streams is None:
        streams = [True] * len(stream_sizes)

    # Straem indices for static+delta features
    # [0,   180, 183, 184]
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    # [180, 183, 184, 199]
    end_indices = np.cumsum(stream_sizes)

    ret = []
    for in_start_idx, in_end_idx, v, enabled in zip(
        start_indices,
        end_indices,
        has_dynamic_features,
        streams,
    ):
        if not enabled:
            continue
        x = inputs[:, in_start_idx:in_end_idx]
        if inputs.shape == variances.shape:
            var_ = variances[:, in_start_idx:in_end_idx]
        else:
            var_ = np.tile(variances[in_start_idx:in_end_idx], (T, 1))
        y = paramgen.mlpg(x, var_, windows) if v else x
        ret.append(y)

    return np.concatenate(ret, -1)
