import torch
from torch import nn
from torch.nn import functional as F
from ttslearn.dsp import mulaw_quantize
from ttslearn.wavenet.modules import Conv1d1x1, ResSkipBlock
from ttslearn.wavenet.upsample import ConvInUpsampleNetwork


class WaveNet(nn.Module):
    """WaveNet

    Args:
        out_channels (int): the number of output channels
        layers (int): the number of layers
        stacks (int): the number of residual stacks
        residual_channels (int): the number of residual channels
        gate_channels (int): the number of channels for the gating function
        skip_out_channels (int): the number of channels in the skip output
        kernel_size (int): the size of the convolutional kernel
        cin_channels (int): the number of input channels for local conditioning
        upsample_scales (list): the list of scales to upsample the local conditioning features
        aux_context_window (int): the number of context frames
    """

    def __init__(
        self,
        out_channels=256,  # 出力のチャネル数
        layers=30,  # レイヤー数
        stacks=3,  # 畳み込みブロックの数
        residual_channels=64,  # 残差結合のチャネル数
        gate_channels=128,  # ゲートのチャネル数
        skip_out_channels=64,  # スキップ接続のチャネル数
        kernel_size=2,  # 1 次元畳み込みのカーネルサイズ
        cin_channels=80,  # 条件付け特徴量のチャネル数
        upsample_scales=None,  # アップサンプリングのスケール
        aux_context_window=0,  # アップサンプリング時に参照する近傍フレーム数
    ):
        super().__init__()
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.aux_context_window = aux_context_window
        if upsample_scales is None:
            upsample_scales = [10, 8]
        self.upsample_scales = upsample_scales

        self.first_conv = Conv1d1x1(out_channels, residual_channels)

        # メインとなる畳み込み層
        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResSkipBlock(
                residual_channels,
                gate_channels,
                kernel_size,
                skip_out_channels,
                dilation=dilation,
                cin_channels=cin_channels,
            )
            self.main_conv_layers.append(conv)

        # スキップ接続の和から波形への変換
        self.last_conv_layers = nn.ModuleList(
            [
                nn.ReLU(),
                Conv1d1x1(skip_out_channels, skip_out_channels),
                nn.ReLU(),
                Conv1d1x1(skip_out_channels, out_channels),
            ]
        )

        # フレーム単位の特徴量をサンプル単位にアップサンプリング
        self.upsample_net = ConvInUpsampleNetwork(
            upsample_scales, cin_channels, aux_context_window
        )

    def forward(self, x, c):
        """Forward step

        Args:
            x (torch.Tensor): the input waveform
            c (torch.Tensor): the local conditioning feature

        Returns:
            torch.Tensor: the output waveform
        """
        # 量子化された離散値列から One-hot ベクトルに変換
        # (B, T) -> (B, T, out_channels) -> (B, out_channels, T)
        x = F.one_hot(x, self.out_channels).transpose(1, 2).float()

        # 条件付き特徴量のアップサンプリング
        c = self.upsample_net(c)
        assert c.size(-1) == x.size(-1)

        # One-hot ベクトルの次元から隠れ層の次元に変換
        x = self.first_conv(x)

        # メインの畳み込み層の処理
        # 各層におけるスキップ接続の出力を加算して保持
        skips = 0
        for f in self.main_conv_layers:
            x, h = f(x, c)
            skips += h

        # スキップ接続の和を入力として、出力を計算
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        # NOTE: 出力を確率値として解釈する場合には softmax が必要ですが、
        # 学習時には nn.CrossEntropyLoss の計算に置いて softmax の計算が行われるので、
        # ここでは明示的に softmax を計算する必要はありません
        return x

    def inference(self, c, num_time_steps=100, tqdm=lambda x: x):
        """Inference step

        Args:
            c (torch.Tensor): the local conditioning feature
            num_time_steps (int): the number of time steps to generate
            tqdm (lambda): a tqdm function to track progress

        Returns:
            torch.Tensor: the output waveform
        """
        self.clear_buffer()

        # Local conditioning
        B = c.shape[0]
        # (B, C, T)
        c = self.upsample_net(c)
        # (B, C, T) -> (B, T, C)
        c = c.transpose(1, 2).contiguous()

        outputs = []

        # 自己回帰生成における初期値
        current_input = torch.zeros(B, 1, self.out_channels).to(c.device)
        current_input[:, :, int(mulaw_quantize(0))] = 1

        if tqdm is None:
            ts = range(num_time_steps)
        else:
            ts = tqdm(range(num_time_steps))

        # 逐次的に生成
        for t in ts:
            # 時刻 t における入力は、時刻 t-1 における出力
            if t > 0:
                current_input = outputs[-1]

            # 時刻 t における条件付け特徴量
            ct = c[:, t, :].unsqueeze(1)

            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = 0
            for f in self.main_conv_layers:
                x, h = f.incremental_forward(x, ct)
                skips += h
            x = skips
            for f in self.last_conv_layers:
                if hasattr(f, "incremental_forward"):
                    x = f.incremental_forward(x)
                else:
                    x = f(x)
            # Softmax によって、出力をカテゴリカル分布のパラメータに変換
            x = F.softmax(x.view(B, -1), dim=1)
            # カテゴリカル分布からサンプリング
            x = torch.distributions.OneHotCategorical(x).sample()
            outputs += [x.data]

        # T x B x C
        # 各時刻における出力を結合
        outputs = torch.stack(outputs)
        # B x C x T
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        """Clear the internal buffer."""
        self.first_conv.clear_buffer()
        for f in self.main_conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def remove_weight_norm_(self):
        """Remove weight normalization of the model"""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)
