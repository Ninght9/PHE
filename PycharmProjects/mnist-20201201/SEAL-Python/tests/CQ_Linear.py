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





def LinearFunction(input, weight, bias=None):
    print("LinearFunction")
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    #print('inpt.dim=',input.dim())
    #print(input.size(),weight.size(),weight.t().size(),bias.size())
    #print(type(input))#, weight.dtype(), weight.t().dtype(), bias.dtype())
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        #1*inputs+1*(inputs_t@inputs_t)  b+xw
        bias_numpy=bias.cpu().detach().numpy()
        input_numpy = input.cpu().detach().numpy()
        weight_t_numpy=weight.t().cpu().detach().numpy()
        weight_numpy = weight.cpu().detach().numpy()

       # bias_int = (bias * 10).type(torch.cuda.IntTensor)  # first *255,then to INT,to Float
       # bias_float = bias_int.type(torch.cuda.FloatTensor)

        #input_int = (input * 10).type(torch.cuda.IntTensor)  # first *255,then to INT,to Float
       # input_float = input_int.type(torch.cuda.FloatTensor)

       # weight_int = (weight * 10).type(torch.cuda.IntTensor)  # first *255,then to INT,to Float
       # weight_float = weight_int.type(torch.cuda.FloatTensor)
      #  ret = torch.addmm(bias.trunc(), input.trunc(), weight.trunc().t())
       #ret = torch.addmm(bias_float, input_float, weight_float.t())
        ret = torch.addmm(bias, input, weight.t())

       # ret_numpy=np.dot(input_numpy,weight_t_numpy)
        ret_numpy=FHEDot(input_numpy,weight_numpy)
        ret_numpy_add = torch.from_numpy(np.add(ret_numpy,bias_numpy)).type(torch.cuda.FloatTensor)

        #print(ret_numpy.size())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
  #  print(ret.size())
   # return ret  #[16,10],so b[16,10]
    return ret_numpy_add



class Linear(torch.nn.Module):
    def __init__(self,input_features,output_features,bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = torch.nn.Parameter(torch.Tensor(output_features,input_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias',None)
        #self.reset_parameters()   # Default initializationb
    def forward(self, input):
        return LinearFunction(input,self.weight,self.bias)