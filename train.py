import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque

#This is all rubbish, we actually need to use Q learning to train the fish to avoid the shark, but for now we will just use a simple neural network to control the fish's movement based on the distance and angle to the shark
