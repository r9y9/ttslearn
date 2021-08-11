# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from torch import nn
from torch.nn import functional as F
from ttslearn.util import make_pad_mask


class BahdanauAttention(nn.Module):
    """Bahdanau-style attention

    This is an attention mechanism originally used in Tacotron.

    Args:
        encoder_dim (int): dimension of encoder outputs
        decoder_dim (int): dimension of decoder outputs
        hidden_dim (int): dimension of hidden state
    """

    def __init__(self, encoder_dim=512, decoder_dim=1024, hidden_dim=128):
        super().__init__()
        self.mlp_enc = nn.Linear(encoder_dim, hidden_dim)
        self.mlp_dec = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, 1)

        self.processed_memory = None

    def reset(self):
        """Reset the internal buffer"""
        self.processed_memory = None

    def forward(
        self,
        encoder_outs,
        src_lens,
        decoder_state,
        mask=None,
    ):
        """Forward step

        Args:
            encoder_outs (torch.FloatTensor): encoder outputs
            src_lens (list): length of each input batch
            decoder_state (torch.FloatTensor): decoder hidden state
            mask (torch.FloatTensor): mask for padding
        """
        # エンコーダに全結合層を適用した結果を保持
        if self.processed_memory is None:
            self.processed_memory = self.mlp_enc(encoder_outs)

        # (B, 1, hidden_dim)
        decoder_state = self.mlp_dec(decoder_state).unsqueeze(1)

        # NOTE: アテンションエネルギーは、デコーダの隠れ状態を入力として、
        # エンコーダの特徴量のみによって決まる
        erg = self.w(torch.tanh(self.processed_memory + decoder_state)).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights


class LocationSensitiveAttention(nn.Module):
    """Location-sensitive attention

    This is an attention mechanism used in Tacotron 2.

    Args:
        encoder_dim (int): dimension of encoder outputs
        decoder_dim (int): dimension of decoder outputs
        hidden_dim (int): dimension of hidden state
        conv_channels (int): number of channels of convolutional layer
        conv_kernel_size (int): size of convolutional kernel
    """

    def __init__(
        self,
        encoder_dim=512,
        decoder_dim=1024,
        hidden_dim=128,
        conv_channels=32,
        conv_kernel_size=31,
    ):
        super().__init__()
        self.mlp_enc = nn.Linear(encoder_dim, hidden_dim)
        self.mlp_dec = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.mlp_att = nn.Linear(conv_channels, hidden_dim, bias=False)
        assert conv_kernel_size % 2 == 1
        self.loc_conv = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )
        self.w = nn.Linear(hidden_dim, 1)

        self.processed_memory = None

    def reset(self):
        """Reset the internal buffer"""
        self.processed_memory = None

    def forward(
        self,
        encoder_outs,
        src_lens,
        decoder_state,
        att_prev,
        mask=None,
    ):
        """Forward step

        Args:
            encoder_outs (torch.FloatTensor): encoder outputs
            src_lens (list): length of each input batch
            decoder_state (torch.FloatTensor): decoder hidden state
            att_prev (torch.FloatTensor): previous attention weight
            mask (torch.FloatTensor): mask for padding
        """
        # エンコーダに全結合層を適用した結果を保持
        if self.processed_memory is None:
            self.processed_memory = self.mlp_enc(encoder_outs)

        # アテンション重みを一様分布で初期化
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(src_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

        # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
        # (B, T_enc, conv_channels)
        att_conv = self.loc_conv(att_prev.unsqueeze(1)).transpose(1, 2)
        # (B, T_enc, hidden_dim)
        att_conv = self.mlp_att(att_conv)

        # (B, 1, hidden_dim)
        decoder_state = self.mlp_dec(decoder_state).unsqueeze(1)

        # NOTE: アテンションエネルギーは、デコーダの隠れ状態を入力として、次の2 つに依存します
        # 1) デコーダの前の時刻におけるアテンション重み
        # 2) エンコーダの隠れ状態
        erg = self.w(
            torch.tanh(att_conv + self.processed_memory + decoder_state)
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights


# 書籍中の数式に沿って、わかりやすさを重視した実装
class BahdanauAttentionNaive(nn.Module):
    def __init__(self, encoder_dim=512, decoder_dim=1024, hidden_dim=128):
        super().__init__()
        self.V = nn.Linear(encoder_dim, hidden_dim)
        self.W = nn.Linear(decoder_dim, hidden_dim, bias=False)
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        encoder_outs,
        decoder_state,
        mask=None,
    ):
        # 式 (9.11) の計算
        erg = self.w(
            torch.tanh(self.W(decoder_state).unsqueeze(1) + self.V(encoder_outs))
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights


# 書籍中の数式に沿って、わかりやすさを重視した実装
class LocationSensitiveAttentionNaive(nn.Module):
    def __init__(
        self,
        encoder_dim=512,
        decoder_dim=1024,
        hidden_dim=128,
        conv_channels=32,
        conv_kernel_size=31,
    ):
        super().__init__()
        self.V = nn.Linear(encoder_dim, hidden_dim)
        self.W = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.U = nn.Linear(conv_channels, hidden_dim, bias=False)
        self.F = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        encoder_outs,
        src_lens,
        decoder_state,
        att_prev,
        mask=None,
    ):
        # アテンション重みを一様分布で初期化
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(src_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

        # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
        # (B, T_enc, conv_channels)
        f = self.F(att_prev.unsqueeze(1)).transpose(1, 2)

        # 式 (9.13) の計算
        erg = self.w(
            torch.tanh(
                self.W(decoder_state).unsqueeze(1) + self.V(encoder_outs) + self.U(f)
            )
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights
