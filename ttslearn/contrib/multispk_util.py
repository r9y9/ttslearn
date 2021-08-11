from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
from ttslearn.logger import getLogger
from ttslearn.train_util import (
    ensure_divisible_by,
    num_trainable_params,
    set_epochs_based_on_max_steps_,
)
from ttslearn.util import init_seed, load_utt_list, pad_1d, pad_2d


class Dataset(data_utils.Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output files
        spk_paths (list): List of paths to speaker ID
    """

    def __init__(self, in_paths, out_paths, spk_paths):
        self.in_paths = in_paths
        self.out_paths = out_paths
        self.spk_paths = spk_paths

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input, target and speaker ID in numpy format
        """
        spk_id = np.load(self.spk_paths[idx])
        return np.load(self.in_paths[idx]), np.load(self.out_paths[idx]), spk_id

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


def collate_fn_ms_tacotron(batch, reduction_factor=1):
    """Collate function for multi-speaker Tacotron.

    Args:
        batch (list): List of tuples of the form (inputs, targets, spk_ids).
        reduction_factor (int, optional): Reduction factor. Defaults to 1.

    Returns:
        tuple: Batch of inputs, input lengths, targets, target lengths, stop flags and spk ids.
    """
    xs = [x[0] for x in batch]
    ys = [ensure_divisible_by(x[1], reduction_factor) for x in batch]
    spk_ids = torch.tensor([int(x[2]) for x in batch], dtype=torch.long).view(-1, 1)
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

    return x_batch, il_batch, y_batch, ol_batch, stop_flags, spk_ids


def get_data_loaders(data_config, collate_fn):
    """Get data loaders for training and validation.

    Args:
        data_config (dict): Data configuration.
        collate_fn (callable): Collate function.

    Returns:
        dict: Data loaders for multi-speaker training.
    """
    data_loaders = {}

    for phase in ["train", "dev"]:
        utt_ids = load_utt_list(to_absolute_path(data_config[phase].utt_list))
        in_dir = Path(to_absolute_path(data_config[phase].in_dir))
        out_dir = Path(to_absolute_path(data_config[phase].out_dir))

        in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_feats_paths = [out_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        spk_id_paths = [in_dir / f"{utt_id}-spk.npy" for utt_id in utt_ids]

        dataset = Dataset(in_feats_paths, out_feats_paths, spk_id_paths)
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,
            shuffle=phase.startswith("train"),
        )

    return data_loaders


def setup(config, device, collate_fn):
    """Setup for traiining

    Args:
        config (dict): configuration for training
        device (torch.device): device to use for training
        collate_fn (callable): function to collate mini-batches

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, and logger.
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
        state_dict = checkpoint["state_dict"]
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        invalid_keys = []
        for k, v in state_dict.items():
            if model_dict[k].shape != v.shape:
                logger.info(f"Skip loading {k}")
                invalid_keys.append(k)
        for k in invalid_keys:
            state_dict.pop(k)
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

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
