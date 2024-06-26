import os,sys
from src.logger import logging
from src.exception import CustomException

import torch
import torch.nn as nn
import torch.optim as optim
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
from PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    try:
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
            super(Transformer, self).__init__()
            self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
            self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

            self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
            self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

            self.fc = nn.Linear(d_model, tgt_vocab_size)
            self.dropout = nn.Dropout(dropout)

        def generate_mask(self, src, tgt):
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
            seq_length = tgt.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
            tgt_mask = tgt_mask & nopeak_mask
            return src_mask, tgt_mask

        def forward(self, src, tgt):
            logging.info("In Transformer started with Generating Mask")
            src_mask, tgt_mask = self.generate_mask(src, tgt)
            logging.info("In Transformer src and tgt is getting embedded using Positional Encoding")
            src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

            enc_output = src_embedded
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)

            dec_output = tgt_embedded
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

            output = self.fc(dec_output)
            return output
    except Exception as e:
        raise CustomException(e,sys)
    
if __name__ == '__main__':
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step() 
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch: {epoch+1}, Loss: {loss.item()}")