"""
Adapted from code originally written by, 
Sriram Ravindran, sriram@ucsd.edu

Original paper - https://arxiv.org/abs/1611.08024

And accessed through github repository:
https://github.com/aliasvishnu/EEGNet

Subsequent changes made to allow evaluation of model with others on EEG data.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


