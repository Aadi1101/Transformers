import os,sys
from src.logger import logging
from src.exception import CustomException

import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionWiseFeedForward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    try:
        def __init__(self, d_model, num_heads, d_ff, dropout):
            super(EncoderLayer, self).__init__()
            logging.info("In Encoding started with Multi Head Attention")
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            logging.info("In Encoding started with Position Wise Feed Forward")
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
            logging.info("Added 2 Normalization Layer.")
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            logging.info("Added Dropout in Encoding")
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask):
            logging.info("Started with Encoder Layer")
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x
    except Exception as e:
        raise CustomException(e,sys)