# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
#import f_functionalCQ as F
import torch.nn.functional as F
from torch.nn import init

from torch.nn import Module
#from torch.nn import *
from torch.nn.modules.utils import _single, _pair, _triple
#from .utils import _single, _pair, _triple
#from ..._jit_internal import List
from torch._jit_internal import List
import importlib
#importlib.reload(f_functionalCQ)
import numpy as np


def reluCQ(input):
    input = input.cpu().detach().numpy()
    result = input*(input>0)
    result = torch.from_numpy(result).type(torch.cuda.FloatTensor)
    return result


