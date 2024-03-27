import os,sys
from src.logger import logging
from src.exception import CustomException
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    try:
        def __init__(self, d_model, max_seq_length):
            super(PositionalEncoding, self).__init__()
            
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
            
        def forward(self, x):
            logging.info(f"In Positional Encoding each element in the input tensor is added to the corresponding positional encoding value, with the positional encodings adjusted to match the length of the input sequence")
            logging.info("Completed with Positional Encoding")
            return x + self.pe[:, :x.size(1)]
    
    except Exception as e:
        raise CustomException(e,sys)