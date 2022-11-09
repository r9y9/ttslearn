from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn
from ttslearn.train_util import (
    collate_fn_dnntts,
    get_epochs_with_optional_tqdm,
    save_checkpoint,
    setup,
)
from ttslearn.util import make_non_pad_mask


def train_step(model, optimizer, train, in_feats, out_feats, lengths):
    optimizer.zero_grad()

    # 順伝播
    pred_out_feats = model(in_feats, lengths)

    # ゼロパディングされた部分を損失の計算に含めないように、マスクを作成
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)
    pred_out_feats = pred_out_feats.masked_select(mask)
    out_feats = out_feats.masked_select(mask)

    # 損失の計算
    loss = nn.MSELoss()(pred_out_feats, out_feats)

    # 逆伝播、モデルパラメータの更新
    if train:
        loss.backward()
        optimizer.step()

    return loss


def train_loop(
    config, logger, device, model, optimizer, lr_scheduler, data_loaders, writer
):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_loss = torch.finfo(torch.float32).max

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, config.train.nepochs):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            for in_feats, out_feats, lengths in data_loaders[phase]:
                # NOTE: pytorch の PackedSequence の仕様に合わせるため、系列長の降順にソート
                lengths, indices = torch.sort(lengths, dim=0, descending=True)
                in_feats, out_feats = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                )
                loss = train_step(model, optimizer, train, in_feats, out_feats, lengths)
                running_loss += loss.item()
            ave_loss = running_loss / len(data_loaders[phase])
            writer.add_scalar(f"Loss/{phase}", ave_loss, epoch)
            if not train and ave_loss < best_loss:
                best_loss = ave_loss
                save_checkpoint(logger, out_dir, model, optimizer, epoch, is_best=True)

        lr_scheduler.step()
        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(logger, out_dir, model, optimizer, epoch, is_best=False)

    save_checkpoint(logger, out_dir, model, optimizer, config.train.nepochs)


@hydra.main(config_path="conf/train_dnntts", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, lr_scheduler, data_loaders, writer, logger = setup(
        config, device, collate_fn_dnntts
    )
    train_loop(
        config, logger, device, model, optimizer, lr_scheduler, data_loaders, writer
    )


if __name__ == "__main__":
    my_app()
