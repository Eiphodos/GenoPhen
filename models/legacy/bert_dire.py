import torch
from torch import nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 key_size=768, query_size=768, value_size=768):
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"{i}",
                d2l.TransformerEncoderBlock(key_size=key_size, query_size=query_size, value_size=value_size,
                                 num_hiddens=num_hiddens, norm_shape=norm_shape, ffn_num_input=ffn_num_input,
                                 ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads, dropout=dropout,
                                 use_bias=True))

    def forward(self, tokens, x_valid_len_to):
        x = self.token_embedding(tokens)
        for blk in self.blks:
            x = blk(x, x_valid_len_to)
        return x


class AbPred(nn.Module):

    def __init__(self, num_hiddens):
        super(AbPred, self).__init__()
        self.ab = nn.Sequential(nn.Linear(num_hiddens, num_hiddens),
                                nn.ReLU(), nn.LayerNorm(num_hiddens),
                                nn.Linear(num_hiddens, 2))

    def forward(self, x):
        return self.ab(x)


class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, attention_heads,
                 attention_layers, dropout, key_size=768, query_size=768, value_size=768, number_ab=16):
        super(BERTModel, self).__init__()

        self.encoder = BERTEncoder(vocab_size=vocab_size, num_hiddens=num_hiddens, norm_shape=norm_shape,
                                   ffn_num_input=ffn_num_input, ffn_num_hiddens=ffn_num_hiddens,
                                   num_heads=attention_heads, num_layers=attention_layers, dropout=dropout,
                                   key_size=key_size, query_size=query_size, value_size=value_size)
        self.number_ab = number_ab
        self.abpred = [AbPred(num_hiddens) for _ in range(self.number_ab)]

    def forward(self, x, positions_y, batch_index_y, positions_x, batch_index_x, x_valid_len_to):
        encoded_x = self.encoder(x, x_valid_len_to)
        mlm_hat = torch.cat([a(encoded_x[:, 0:1, :]) for a in self.abpred], 1)
        ab_y_hat = mlm_hat[batch_index_y, positions_y]
        ab_x_hat = mlm_hat[batch_index_x, positions_x]

        return ab_y_hat, ab_x_hat

    def to_within(self, device):
        for i in range(self.number_ab):
            self.abpred[i].to(device)
