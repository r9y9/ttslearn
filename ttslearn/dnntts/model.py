from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DNN(nn.Module):
    """Feed-forward neural network

    Args:
        in_dim: input dimension
        hidden_dim: hidden dimension
        out_dim: output dimension
        num_layers: number of layers
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super(DNN, self).__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(nn.ReLU())
        model.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*model)

    def forward(self, seqs, lens=None):
        """Forward step

        Args:
            seqs (torch.Tensor): input sequences
            lens (torch.Tensor): length of input sequences

        Returns:
            torch.Tensor: output sequences
        """
        return self.model(seqs)


class LSTMRNN(nn.Module):
    """LSTM-based recurrent neural networks

    Args:
        in_dim (int): input dimension
        hidden_dim (int): hidden dimension
        out_dim (int): output dimension
        num_layers (int): number of layers
        bidirectional (bool): bi-directional or not.
        dropout (float): dropout ratio.
    """

    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True, dropout=0.0
    ):
        super(LSTMRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.hidden2out = nn.Linear(self.num_direction * self.hidden_dim, out_dim)

    def forward(self, seqs, lens):
        """Forward step

        Args:
            seqs (torch.Tensor): input sequences
            lens (torch.Tensor): length of input sequences

        Returns:
            torch.Tensor: output sequences
        """
        seqs = pack_padded_sequence(seqs, lens, batch_first=True)
        out, _ = self.lstm(seqs)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out
