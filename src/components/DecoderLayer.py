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
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.cross_attn = MultiHeadAttention(d_model, num_heads)
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, enc_output, src_mask, tgt_mask):
            logging.info("In Decoder Layer started with Self Multi Head Attention.")
            attn_output = self.self_attn(x, x, x, tgt_mask)
            logging.info("In Decoder Layer added 1st Normalization Layer")
            x = self.norm1(x + self.dropout(attn_output))
            logging.info("In Decoder layer added Cross Multi Head Attention")
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
            logging.info("In Decoder layer added 2nd Normalization Layer")
            x = self.norm2(x + self.dropout(attn_output))
            logging.info("In Decoder layer started with Position Wise Feed Forward")
            ff_output = self.feed_forward(x)
            logging.info("In Decoder layer added 3rd Normalization layer")
            x = self.norm3(x + self.dropout(ff_output))
            logging.info("Completed with Decoder Layer")
            return x
    except Exception as e:
        raise CustomException(e,sys)