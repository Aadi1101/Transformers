import sys,os
from src.logger import logging
from src.exception import CustomException

import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    try:
        def __init__(self, d_model, num_heads, d_ff, dropout):
            super(DecoderLayer, self).__init__()
            logging.info("In Decoding started with Multi Head Attention with self attention and cross attention")
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.cross_attn = MultiHeadAttention(d_model, num_heads)
            logging.info("In Decoding started with PositionWiseFeedForward")
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
            logging.info("In Decoding added 3 Normalization Layer")
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            logging.info("In Decoding added the Dropout layer")
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, enc_output, src_mask, tgt_mask):
            logging.info("Started with Decoder Layer.")
            attn_output = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(attn_output))
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout(ff_output))
            return x
    except Exception as e:
        raise CustomException(e,sys)