import shutil
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
from ttslearn.logger import getLogger
from ttslearn.util import init_seed, load_utt_list, pad_1d, pad_2d


def get_epochs_with_optional_tqdm(tqdm_mode, nepochs):
    """Get epochs with optional progress bar.

    Args:
        tqdm_mode (str): Progress bar mode.
        nepochs (int): Number of epochs.

    Returns:
        iterable: Epochs.
    """
    if tqdm_mode == "tqdm":
        from tqdm import tqdm

        epochs = tqdm(range(1, nepochs + 1), desc="epoch")
    else:
        epochs = range(1, nepochs + 1)

    return epochs


def moving_average_(model, model_test, beta=0.9999):
    """Exponential moving average (EMA) of model parameters.

    Args:
        model (torch.nn.Module): Model to perform EMA on.
        model_test (torch.nn.Module): Model to use for the test phase.
        beta (float, optional): [description]. Defaults to 0.9999.
    """
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def num_trainable_params(model):
    """Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): Model to count the number of trainable parameters.

    Returns:
        int: Number of trainable parameters.
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


class Dataset(data_utils.Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output files
    """

    def __init__(self, in_paths, out_paths):
        self.in_paths = in_paths
        self.out_paths = out_paths

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return np.load(self.in_paths[idx]), np.load(self.out_paths[idx])

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


def get_data_loaders(data_config, collate_fn):
    """Get data loaders for training and validation.

    Args:
        data_config (dict): Data configuration.
        collate_fn (callable): Collate function.

    Returns:
        dict: Data loaders.
    """
    data_loaders = {}

    for phase in ["train", "dev"]:
        utt_ids = load_utt_list(to_absolute_path(data_config[phase].utt_list))
        in_dir = Path(to_absolute_path(data_config[phase].in_dir))
        out_dir = Path(to_absolute_path(data_config[phase].out_dir))

        in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_feats_paths = [out_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        dataset = Dataset(in_feats_paths, out_feats_paths)
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,
            shuffle=phase.startswith("train"),
        )

    return data_loaders


def collate_fn_dnntts(batch):
    """Collate function for DNN-TTS.

    Args:
        batch (list): List of tuples of the form (inputs, targets).

    Returns:
        tuple: Batch of inputs, targets, and lengths.
    """
    lengths = [len(x[0]) for x in batch]
    max_len = max(lengths)
    x_batch = torch.stack([torch.from_numpy(pad_2d(x[0], max_len)) for x in batch])
    y_batch = torch.stack([torch.from_numpy(pad_2d(x[1], max_len)) for x in batch])
    l_batch = torch.tensor(lengths, dtype=torch.long)
    return x_batch, y_batch, l_batch


def collate_fn_wavenet(batch, max_time_frames=100, hop_size=80, aux_context_window=2):
    """Collate function for WaveNet.

    Args:
        batch (list): List of tuples of the form (inputs, targets).
        max_time_frames (int, optional): Number of time frames. Defaults to 100.
        hop_size (int, optional): Hop size. Defaults to 80.
        aux_context_window (int, optional): Auxiliary context window. Defaults to 2.

    Returns:
        tuple: Batch of waveforms and conditional features.
    """
    max_time_steps = max_time_frames * hop_size

    xs, cs = [b[1] for b in batch], [b[0] for b in batch]

    # 条件付け特徴量の開始位置をランダム抽出した後、それに相当する短い音声波形を切り出します
    c_lengths = [len(c) for c in cs]
    start_frames = np.array(
        [
            np.random.randint(
                aux_context_window, cl - aux_context_window - max_time_frames
            )
            for cl in c_lengths
        ]
    )
    x_starts = start_frames * hop_size
    x_ends = x_starts + max_time_steps
    c_starts = start_frames - aux_context_window
    c_ends = start_frames + max_time_frames + aux_context_window
    x_cut = [x[s:e] for x, s, e in zip(xs, x_starts, x_ends)]
    c_cut = [c[s:e] for c, s, e in zip(cs, c_starts, c_ends)]

    # numpy.ndarray のリスト型から torch.Tensor 型に変換します
    x_batch = torch.tensor(x_cut, dtype=torch.long)  # (B, T)
    c_batch = torch.tensor(c_cut, dtype=torch.float).transpose(2, 1)  # (B, C, T')

    return x_batch, c_batch


def ensure_divisible_by(feats, N):
    """Ensure that the number of frames is divisible by N.

    Args:
        feats (np.ndarray): Input features.
        N (int): Target number of frames.

    Returns:
        np.ndarray: Input features with number of frames divisible by N.
    """
    if N == 1:
        return feats
    mod = len(feats) % N
    if mod != 0:
        feats = feats[: len(feats) - mod]
    return feats


def collate_fn_tacotron(batch, reduction_factor=1):
    """Collate function for Tacotron.

    Args:
        batch (list): List of tuples of the form (inputs, targets).
        reduction_factor (int, optional): Reduction factor. Defaults to 1.

    Returns:
        tuple: Batch of inputs, input lengths, targets, target lengths and stop flags.
    """
    xs = [x[0] for x in batch]
    ys = [ensure_divisible_by(x[1], reduction_factor) for x in batch]
    in_lens = [len(x) for x in xs]
    out_lens = [len(y) for y in ys]
    in_max_len = max(in_lens)
    out_max_len = max(out_lens)
    x_batch = torch.stack([torch.from_numpy(pad_1d(x, in_max_len)) for x in xs])
    y_batch = torch.stack([torch.from_numpy(pad_2d(y, out_max_len)) for y in ys])
    il_batch = torch.tensor(in_lens, dtype=torch.long)
    ol_batch = torch.tensor(out_lens, dtype=torch.long)
    stop_flags = torch.zeros(y_batch.shape[0], y_batch.shape[1])
    for idx, out_len in enumerate(out_lens):
        stop_flags[idx, out_len - 1 :] = 1.0
    return x_batch, il_batch, y_batch, ol_batch, stop_flags


def set_epochs_based_on_max_steps_(train_config, steps_per_epoch, logger):
    """Set epochs based on max steps.

    Args:
        train_config (TrainConfig): Train config.
        steps_per_epoch (int): Number of steps per epoch.
        logger (logging.Logger): Logger.
    """
    logger.info(f"Number of iterations per epoch: {steps_per_epoch}")

    if train_config.max_train_steps < 0:
        # Set max_train_steps based on nepochs
        max_train_steps = train_config.nepochs * steps_per_epoch
        train_config.max_train_steps = max_train_steps
        logger.info(
            "Number of max_train_steps is set based on nepochs: {}".format(
                max_train_steps
            )
        )
    else:
        # Set nepochs based on max_train_steps
        max_train_steps = train_config.max_train_steps
        epochs = int(np.ceil(max_train_steps / steps_per_epoch))
        train_config.nepochs = epochs
        logger.info(
            "Number of epochs is set based on max_train_steps: {}".format(epochs)
        )

    logger.info(f"Number of epochs: {train_config.nepochs}")
    logger.info(f"Number of iterations: {train_config.max_train_steps}")


def save_checkpoint(
    logger, out_dir, model, optimizer, epoch, is_best=False, postfix=""
):
    """Save a checkpoint.

    Args:
        logger (logging.Logger): Logger.
        out_dir (str): Output directory.
        model (nn.Module): Model.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch.
        is_best (bool, optional): Whether or not the current model is the best.
            Defaults to False.
        postfix (str, optional): Postfix. Defaults to "".
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_best:
        path = out_dir / f"best_loss{postfix}.pth"
    else:
        path = out_dir / "epoch{:04d}{}.pth".format(epoch, postfix)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

    logger.info(f"Saved checkpoint at {path}")
    if not is_best:
        shutil.copyfile(path, out_dir / f"latest{postfix}.pth")


def plot_attention(alignment):
    """Plot attention.

    Args:
        alignment (np.ndarray): Attention.
    """
    fig, ax = plt.subplots()
    alignment = alignment.cpu().data.numpy().T
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    plt.xlabel("Decoder time step")
    plt.ylabel("Encoder time step")
    return fig


def plot_2d_feats(feats, title=None):
    """Plot 2D features.

    Args:
        feats (np.ndarray): Input features.
        title (str, optional): Title. Defaults to None.
    """
    feats = feats.cpu().data.numpy().T
    fig, ax = plt.subplots()
    im = ax.imshow(
        feats, aspect="auto", origin="lower", interpolation="none", cmap="viridis"
    )
    fig.colorbar(im, ax=ax)
    if title is not None:
        ax.set_title(title)
    return fig


def setup(config, device, collate_fn):
    """Setup for traiining

    Args:
        config (dict): configuration for training
        device (torch.device): device to use for training
        collate_fn (callable): function to collate mini-batches

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, and logger.

    .. note::

        書籍に記載のコードは、この関数を一部簡略化しています。
    """
    # NOTE: hydra は内部で stream logger を追加するので、二重に追加しないことに注意
    logger = getLogger(config.verbose, add_stream_handler=False)

    logger.info(f"PyTorch version: {torch.__version__}")

    # CUDA 周りの設定
    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.cudnn.benchmark
        cudnn.deterministic = config.cudnn.deterministic
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")
    init_seed(config.seed)

    # モデルのインスタンス化
    model = hydra.utils.instantiate(config.model.netG).to(device)
    logger.info(model)
    logger.info(
        "Number of trainable params: {:.3f} million".format(
            num_trainable_params(model) / 1000000.0
        )
    )

    # (optional) 学習済みモデルの読み込み
    # ファインチューニングしたい場合
    pretrained_checkpoint = config.train.pretrained.checkpoint
    if pretrained_checkpoint is not None and len(pretrained_checkpoint) > 0:
        logger.info(
            "Fine-tuning! Loading a checkpoint: {}".format(pretrained_checkpoint)
        )
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # 複数 GPU 対応
    if config.data_parallel:
        model = nn.DataParallel(model)

    # Optimizer
    optimizer_class = getattr(optim, config.train.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **config.train.optim.optimizer.params
    )

    # 学習率スケジューラ
    lr_scheduler_class = getattr(
        optim.lr_scheduler, config.train.optim.lr_scheduler.name
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **config.train.optim.lr_scheduler.params
    )

    # DataLoader
    data_loaders = get_data_loaders(config.data, collate_fn)

    set_epochs_based_on_max_steps_(config.train, len(data_loaders["train"]), logger)

    # Tensorboard の設定
    writer = SummaryWriter(to_absolute_path(config.train.log_dir))

    # config ファイルを保存しておく
    out_dir = Path(to_absolute_path(config.train.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.yaml", "w") as f:
        OmegaConf.save(config.model, f)
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    return model, optimizer, lr_scheduler, data_loaders, writer, logger
