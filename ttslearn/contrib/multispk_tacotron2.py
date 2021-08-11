import torch
from torch import nn
from ttslearn.tacotron.decoder import Decoder
from ttslearn.tacotron.encoder import Encoder
from ttslearn.tacotron.postnet import Postnet


class MultiSpkTacotron2(nn.Module):
    """Multi-speaker Tacotron 2

    This implementation does not include the WaveNet vocoder of the Tacotron 2.

    Args:
        num_vocab (int): the size of vocabulary
        embed_dim (int): dimension of embedding
        encoder_hidden_dim (int): dimension of hidden unit
        encoder_conv_layers (int): the number of convolution layers
        encoder_conv_channels (int): the number of convolution channels
        encoder_conv_kernel_size (int): kernel size of convolution
        encoder_dropout (float): dropout rate of convolution
        attention_hidden_dim (int): dimension of hidden unit
        attention_conv_channels (int): the number of convolution channels
        attention_conv_kernel_size (int): kernel size of convolution
        decoder_out_dim (int): dimension of output
        decoder_layers (int): the number of decoder layers
        decoder_hidden_dim (int): dimension of hidden unit
        decoder_prenet_layers (int): the number of prenet layers
        decoder_prenet_hidden_dim (int): dimension of hidden unit
        decoder_prenet_dropout (float): dropout rate of prenet
        decoder_zoneout (float): zoneout rate
        postnet_layers (int): the number of postnet layers
        postnet_channels (int): the number of postnet channels
        postnet_kernel_size (int): kernel size of postnet
        postnet_dropout (float): dropout rate of postnet
        reduction_factor (int): reduction factor
        n_spks (int): the number of speakers
        spk_emb_dim (int): dimension of speaker embedding
    """

    def __init__(
        self,
        num_vocab=51,
        embed_dim=512,
        encoder_hidden_dim=512,
        encoder_conv_layers=3,
        encoder_conv_channels=512,
        encoder_conv_kernel_size=5,
        encoder_dropout=0.5,
        attention_hidden_dim=128,
        attention_conv_channels=32,
        attention_conv_kernel_size=31,
        decoder_out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        decoder_prenet_layers=2,
        decoder_prenet_hidden_dim=256,
        decoder_prenet_dropout=0.5,
        decoder_zoneout=0.1,
        postnet_layers=5,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,
        reduction_factor=1,
        n_spks=100,
        spk_emb_dim=64,
    ):
        super().__init__()
        self.spk_embed = nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = Encoder(
            num_vocab,
            embed_dim,
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.decoder = Decoder(
            encoder_hidden_dim + spk_emb_dim,
            decoder_out_dim,
            decoder_layers,
            decoder_hidden_dim,
            decoder_prenet_layers,
            decoder_prenet_hidden_dim,
            decoder_prenet_dropout,
            decoder_zoneout,
            reduction_factor,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.postnet = Postnet(
            decoder_out_dim,
            postnet_layers,
            postnet_channels,
            postnet_kernel_size,
            postnet_dropout,
        )

    def forward(self, seq, in_lens, decoder_targets, spk_ids):
        """Forward step

        Args:
            seq (torch.Tensor): input sequence
            in_lens (torch.Tensor): input sequence lengths
            decoder_targets (torch.Tensor): target sequence
            spk_ids (torch.Tensor): speaker ids

        Returns:
            tuple: tuple of outputs, outputs (after post-net), stop token prediction
                and attention weights.
        """
        # エンコーダによるテキストの潜在表現の獲得
        encoder_outs = self.encoder(seq, in_lens)

        # 話者埋め込み
        spk_emb = self.spk_embed(spk_ids)
        # 話者埋め込みを時間方向にexpandする
        spk_emb = spk_emb.expand(spk_emb.shape[0], seq.shape[1], spk_emb.shape[-1])

        # 話者埋め込みとエンコーダの出力を結合
        decoder_inp = torch.cat((encoder_outs, spk_emb), dim=-1)

        # デコーダによるメルスペクトログラム、stop token の予測
        outs, logits, att_ws = self.decoder(decoder_inp, in_lens, decoder_targets)

        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return outs, outs_fine, logits, att_ws

    def inference(self, seq, spk_id):
        """Inference step

        Args:
            seq (torch.Tensor): input sequence
            spk_id (int): speaker id

        Returns:
            tuple: tuple of outputs, outputs (after post-net), stop token prediction
                and attention weights.
        """
        seq = seq.unsqueeze(0) if len(seq.shape) == 1 else seq
        in_lens = torch.tensor([seq.shape[-1]], dtype=torch.long, device=seq.device)
        spk_id = spk_id.unsqueeze(0) if len(spk_id.shape) == 1 else spk_id

        outs, outs_fine, logits, att_ws = self.forward(seq, in_lens, None, spk_id)

        return outs[0], outs_fine[0], logits[0], att_ws[0]
