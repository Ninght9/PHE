import torch
import torch.nn.functional as F
#import f_functionalCQ as F
#import f_functionalCQ as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import numpy as np


import time
import math
import random
from seal import *
from seal_helper import *
from importlib import reload

from f_FHEDot import *





def SigmoidFunction(input):
    print("SigmoidFunction")
    output=1.0/(1+np.exp(-(float)(X)))
    return output



class SigmoidCQ(torch.nn.Module):
    def __init__(self):
        super(SigmoidCQ, self).__init__()

    def forward(input):
        return SigmoidFunction(input)