import os,sys
from src.logger import logging
from src.exception import CustomException
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    try:
        def __init__(self, d_model, d_ff):
            super(PositionWiseFeedForward, self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.relu = nn.ReLU()
            

        def forward(self, x):
            logging.info('Position Wise Feed Forward starts with 1st dense layer')
            logging.info('In Position Wise Feed Forward passed through ReLU activation function')
            logging.info('In Position Wise Feed Forward added 2nd dense layer')
            logging.info("Completed with Position Wise Feed Forward")
            return self.fc2(self.relu(self.fc1(x)))
    
    except Exception as e:
        raise CustomException(e,sys)