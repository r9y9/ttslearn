from functools import partial
from logging import Logger
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm
from ttslearn.tacotron.frontend.openjtalk import sequence_to_text
from ttslearn.train_util import (
    collate_fn_tacotron,
    get_epochs_with_optional_tqdm,
    plot_2d_feats,
    plot_attention,
    save_checkpoint,
    setup,
)
from ttslearn.util import make_non_pad_mask

logger: Logger = None


@torch.no_grad()
def eval_model(
    step, model, writer, in_feats, in_lens, out_feats, out_lens, is_inference
):
    # 最大3つまで
    N = min(len(in_feats), 3)

    if is_inference:
        outs, outs_fine, att_ws, out_lens = [], [], [], []
        for idx in range(N):
            out, out_fine, _, att_w = model.inference(in_feats[idx][: in_lens[idx]])
            outs.append(out)
            outs_fine.append(out_fine)
            att_ws.append(att_w)
            out_lens.append(len(out))
    else:
        outs, outs_fine, _, att_ws = model(in_feats, in_lens, out_feats)

    for idx in range(N):
        text = "".join(
            sequence_to_text(in_feats[idx][: in_lens[idx]].cpu().data.numpy())
        )
        if is_inference:
            group = f"utt{idx+1}_inference"
        else:
            group = f"utt{idx+1}_teacher_forcing"

        out = outs[idx][: out_lens[idx]]
        out_fine = outs_fine[idx][: out_lens[idx]]
        rf = model.decoder.reduction_factor
        att_w = att_ws[idx][: out_lens[idx] // rf, : in_lens[idx]]
        fig = plot_attention(att_w)
        writer.add_figure(f"{group}/attention", fig, step)
        plt.close()
        fig = plot_2d_feats(out, text)
        writer.add_figure(f"{group}/out_before_postnet", fig, step)
        plt.close()
        fig = plot_2d_feats(out_fine, text)
        writer.add_figure(f"{group}/out_after_postnet", fig, step)
        plt.close()
        if not is_inference:
            out_gt = out_feats[idx][: out_lens[idx]]
            fig = plot_2d_feats(out_gt, text)
            writer.add_figure(f"{group}/out_ground_truth", fig, step)
            plt.close()


def train_step(
    model,
    optimizer,
    lr_scheduler,
    train,
    criterions,
    in_feats,
    in_lens,
    out_feats,
    out_lens,
    stop_flags,
):
    optimizer.zero_grad()

    # Run forwaard
    outs, outs_fine, logits, _ = model(in_feats, in_lens, out_feats)

    # Mask (B x T x 1)
    mask = make_non_pad_mask(out_lens).unsqueeze(-1).to(out_feats.device)
    out_feats = out_feats.masked_select(mask)
    outs = outs.masked_select(mask)
    outs_fine = outs_fine.masked_select(mask)
    stop_flags = stop_flags.masked_select(mask.squeeze(-1))
    logits = logits.masked_select(mask.squeeze(-1))

    # Loss
    decoder_out_loss = criterions["out_loss"](outs, out_feats)
    postnet_out_loss = criterions["out_loss"](outs_fine, out_feats)
    stop_token_loss = criterions["stop_token_loss"](logits, stop_flags)
    loss = decoder_out_loss + postnet_out_loss + stop_token_loss

    loss_values = {
        "DecoderOutLoss": decoder_out_loss.item(),
        "PostnetOutLoss": postnet_out_loss.item(),
        "StopTokenLoss": stop_token_loss.item(),
        "Loss": loss.item(),
    }

    # Update
    if train:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            logger.info("grad norm is NaN. Skip updating")
        else:
            optimizer.step()
        lr_scheduler.step()

    return loss_values


def _update_running_losses_(running_losses, loss_values):
    for key, val in loss_values.items():
        try:
            running_losses[key] += val
        except KeyError:
            running_losses[key] = val


def train_loop(config, device, model, optimizer, lr_scheduler, data_loaders, writer):
    criterions = {
        "out_loss": nn.MSELoss(),
        "stop_token_loss": nn.BCEWithLogitsLoss(),
    }

    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_loss = torch.finfo(torch.float32).max
    train_iter = 1
    nepochs = config.train.nepochs

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, nepochs):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_losses = {}
            for idx, (in_feats, in_lens, out_feats, out_lens, stop_flags) in tqdm(
                enumerate(data_loaders[phase]), desc=f"{phase} iter", leave=False
            ):
                # ミニバッチのソート
                in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
                in_feats, out_feats, out_lens = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                    out_lens[indices].to(device),
                )
                stop_flags = stop_flags[indices].to(device)

                loss_values = train_step(
                    model,
                    optimizer,
                    lr_scheduler,
                    train,
                    criterions,
                    in_feats,
                    in_lens,
                    out_feats,
                    out_lens,
                    stop_flags,
                )
                if train:
                    for key, val in loss_values.items():
                        writer.add_scalar(f"{key}ByStep/train", val, train_iter)
                    writer.add_scalar(
                        "LearningRate", lr_scheduler.get_last_lr()[0], train_iter
                    )
                    train_iter += 1
                _update_running_losses_(running_losses, loss_values)

                # 最初の検証用データに対して、中間結果の可視化
                if (
                    not train
                    and idx == 0
                    and epoch % config.train.eval_epoch_interval == 0
                ):
                    for is_inference in [False, True]:
                        eval_model(
                            train_iter,
                            model,
                            writer,
                            in_feats,
                            in_lens,
                            out_feats,
                            out_lens,
                            is_inference,
                        )

            # Epoch ごとのロスを出力
            for key, val in running_losses.items():
                ave_loss = val / len(data_loaders[phase])
                writer.add_scalar(f"{key}/{phase}", ave_loss, epoch)

            ave_loss = running_losses["Loss"] / len(data_loaders[phase])
            if not train and ave_loss < best_loss:
                best_loss = ave_loss
                save_checkpoint(logger, out_dir, model, optimizer, epoch, True)

        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(logger, out_dir, model, optimizer, epoch, False)

    # save at last epoch
    save_checkpoint(logger, out_dir, model, optimizer, nepochs)
    logger.info(f"The best loss was {best_loss}")

    return model


@hydra.main(config_path="conf/train_tacotron", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate_fn = partial(
        collate_fn_tacotron, reduction_factor=config.model.netG.reduction_factor
    )
    model, optimizer, lr_scheduler, data_loaders, writer, logger = setup(
        config, device, collate_fn
    )
    train_loop(config, device, model, optimizer, lr_scheduler, data_loaders, writer)


if __name__ == "__main__":
    my_app()
