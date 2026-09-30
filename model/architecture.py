import random
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev, c_prev):
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        g_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


# Name intentionally follows the original notebook for compatibility.
class AdditiveAttetion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        _, src_len, _ = encoder_outputs.shape
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_s(decoder_hidden) + self.W_h(encoder_outputs))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weight = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weight


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            LSTMCell(embed_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.backward_layers = nn.ModuleList([
            LSTMCell(embed_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        batch_size, seq_len = src.shape
        device = src.device
        embedded = self.dropout(self.embedding(src))
        hidden = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        cell = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        hidden_b = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        cell_b = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
        encoder_outputs_f, encoder_outputs_b = [], []

        for t in range(seq_len):
            x = embedded[:, t, :]
            for l in range(self.num_layers):
                inp = x if l == 0 else hidden[l - 1]
                hidden[l], cell[l] = self.layers[l](inp, hidden[l], cell[l])
            encoder_outputs_f.append(hidden[-1])

        for t in reversed(range(seq_len)):
            x = embedded[:, t, :]
            for l in range(self.num_layers):
                inp = x if l == 0 else hidden_b[l - 1]
                hidden_b[l], cell_b[l] = self.backward_layers[l](inp, hidden_b[l], cell_b[l])
            encoder_outputs_b.append(hidden_b[-1])

        encoder_outputs_b.reverse()
        encoder_outputs = torch.stack([
            torch.cat((f, b), dim=1) for f, b in zip(encoder_outputs_f, encoder_outputs_b)
        ], dim=1)
        hidden = [torch.cat((hidden[l], hidden_b[l]), dim=1) for l in range(self.num_layers)]
        cell = [torch.cat((cell[l], cell_b[l]), dim=1) for l in range(self.num_layers)]
        return encoder_outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, pad_idx, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            LSTMCell(embed_dim + hidden_size * 2 if i == 0 else hidden_size * 3, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttetion(hidden_size)
        self.fc_out = nn.Linear(hidden_size + hidden_size * 2, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)).squeeze(1))
        context, _ = self.attention(hidden[-1], encoder_outputs, mask)
        x = torch.cat((embedded, context), dim=1)
        new_hidden, new_cell = [], []
        h, c = self.layers[0](x, hidden[0], cell[0])
        new_hidden.append(h); new_cell.append(c)
        for l in range(1, self.num_layers):
            inp = torch.cat((self.dropout(new_hidden[l - 1]), context), dim=1)
            h, c = self.layers[l](inp, hidden[l], cell[l])
            new_hidden.append(h); new_cell.append(c)
        prediction = self.fc_out(torch.cat((self.dropout(new_hidden[-1]), context), dim=1))
        return prediction, new_hidden, new_cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
        self.bridge_hidden = nn.Linear(encoder.hidden_size * 2, decoder.hidden_size)
        self.bridge_cell = nn.Linear(encoder.hidden_size * 2, decoder.hidden_size)

    def create_mask(self, src):
        return src != self.pad_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = src.shape[0], trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)
        mask = self.create_mask(src)
        encoder_outputs, hidden, cell = self.encoder(src)
        hidden = [torch.tanh(self.bridge_hidden(h)) for h in hidden]
        cell = [torch.tanh(self.bridge_cell(c)) for c in cell]
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs
