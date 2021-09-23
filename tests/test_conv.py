import pytest
import torch
from torch import nn
from ttslearn.wavenet.conv import Conv1d


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 8])
def test_incremental_causal_conv1d(kernel_size, dilation):
    padding = (kernel_size - 1) * dilation
    conv1d = nn.Conv1d(1, 16, kernel_size, padding=padding, dilation=dilation)
    conv1d_inc = Conv1d(1, 16, kernel_size, padding=padding, dilation=dilation)

    # 適当な値でパラメータを初期化
    for c in [conv1d, conv1d_inc]:
        if c.bias is not None:
            nn.init.zeros_(c.bias)
        nn.init.ones_(c.weight.data)
        c.eval()

    B, C, T = 8, 1, 100
    x = torch.zeros(B, C, T) + torch.arange(0, T).float()

    # 1. 各時刻における畳み込みを並列に計算
    with torch.no_grad():
        y1 = conv1d(x)
        y1 = y1[:, :, :T]

    y2 = torch.zeros_like(y1)
    # 2. 逐次的に計算
    for t in range(T):
        # (B, C, 1) -> (B, 1, C)
        xt = x[:, :, t:t+1].transpose(1,2)
        # (B, 1, C)
        with torch.no_grad():
            yt = conv1d_inc.incremental_forward(xt)
        # (B, 1, C) -> (B, C, 1)
        y2[:, :, t:t+1] = yt.transpose(1,2)

    # 並列計算の場合と逐次計算の場合で、計算結果は一致する
    assert (y1 == y2).all()
