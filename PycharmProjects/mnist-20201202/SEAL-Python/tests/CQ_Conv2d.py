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
import math
from seal import *
from seal_helper import *
from f_FHEDot import *

def encrypFHE1d(inputs_b):
    plain_b = Plaintext()
    encrypted_b = Ciphertext(context)
    row_inputs_b = inputs_b.shape[0]  # 8
  #  col_inputs_b = inputs_b.shape[1]  # 1
   # print(row_inputs_b)
#    print(col_inputs_b)
    inputs_b_DoubleVector = DoubleVector()
    for i in range(row_inputs_b):
        inputs_b_DoubleVector.push_back(0)
    for i in range(row_inputs_b):
        inputs_b_DoubleVector[i]=inputs_b[i]
    ckks_encoder.encode(inputs_b_DoubleVector, scale, plain_b)  # bianma
    encryptor.encrypt(plain_b, encrypted_b)
    return encrypted_b,row_inputs_b

def FHEconv2dCQ(input,weight,bias):
    plain_x = Plaintext()
    plain_w = Plaintext()
    plain_b = Plaintext()
    encrypted_x = Ciphertext(context)
    encrypted_w = Ciphertext(context)
    encrypted_b = Ciphertext(context)
    N_inputs_x, C_inputs_x,C_inputs_x,C_inputs_x = input.shape  # 63,1,28,28
    N_inputs_w, C_inputs_w,C_inputs_w,C_inputs_w = input.shape  # 63,1,3,3
    C_inputs_b = inputs_b.shape[0]  # 8

    #  col_inputs_b = inputs_b.shape[1]  # 1
    # print(row_inputs_b)
    #    print(col_inputs_b)
    inputs_x_DoubleVector = DoubleVector()
    inputs_w_DoubleVector = DoubleVector()
    inputs_b_DoubleVector = DoubleVector()

    for i in range(row_inputs_b):
        inputs_b_DoubleVector.push_back(0)
    for i in range(row_inputs_b):
        inputs_b_DoubleVector[i] = inputs_b[i]
    ckks_encoder.encode(inputs_b_DoubleVector, scale, plain_b)  # bianma
    encryptor.encrypt(plain_b, encrypted_b)

    out = None
    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape
    S = 1
    P = 0
    Ho = 1 + (H + 2 * P - HH) / S
    Wo = 1 + (W + 2 * P - WW) / S
    x_pad = np.zeros((N, C, H + 2 * P, W + 2 * P))  # (64,1,28,28) #
    x_pad[:, :, P:P + H, P:P + W] = input  # same as input
    out = np.zeros((N, F, int(Ho), int(Wo)))  # (64,8,26,26) 0000000000
    for f in range(F):
        for i in range(int(Ho)):
            for j in range(int(Wo)):

                a = x_pad[:, :, i * S:i * S + HH, j * S:j * S + WW]  # ()
                b = weight[f, :, :, :]
                c = np.sum(a * b)
                out[:, f, i, j] = np.sum(a * b, axis=(1, 2, 3))
        out[:, f, :, :] += bias[f]


def conv2dCQ4(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()


    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape

    weight2= weight.reshape(1,-1)
    input2=input.reshape((10,784))
    weight3= weight.reshape(F,HH*WW)
    input3=np.zeros((10,HH*WW,H*W)) #9,784
    output=np.zeros((10,8,784))
    print("da")
    for k in range(10-1):
        for i in range(8):
            for j in range(783):
              #  print(i,j,k,input2[0][k-1])
                input3[k][i][j]=input2[k][j]
 #   conv = FHEconv2dCQ(input,weight,bias)
    for i in range(10-1):
       # output[i]=np.dot(weight3,input3[i])
        print("!")
        output[i] = FHEDot(weight3, np.transpose(input3[i]))
    output= output.reshape((10,8,28,28))
    output2=np.zeros(10*8*26*26)
    x=0
    for k in range(10-1):
        for i in range(8-1):
            for j in range(28-1):
                for l in range(28-1):
                    if j == 0 or l == 0 or j==27 or l ==27:
                        break
                    else:
                        output2[x]=output[k][i][j][l]
                        x = x + 1
    output2=output2.reshape((10,8,26*26))


    bias2 = np.zeros((8, 676))
    for i in range(8 - 1):
        for j in range(676 - 1):
            bias2[i][j] = bias[i]
    for i in range(10 - 1):
        output2[i] = np.add(output2[i].reshape(8, 676), bias2)
    output2 = output2.reshape((10, 8, 26, 26))

    conv = torch.from_numpy(output2).type(torch.cuda.FloatTensor)
    #print(weight)
    return conv




def conv2dCQ2(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()


    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape

    weight2= weight.reshape(1,-1)
    input2=input.reshape((10,1,28,28))
    input4=np.zeros(10*1*26*26)
    x = 0
    for k in range(10 - 1):
        for i in range(1 - 1):
            for j in range(28 - 1):
                for l in range(28 - 1):
                    if j == 0 or l == 0 or j == 27 or l == 27:
                        break
                    else:
                        input4[x] = input2[k][i][j][l]
                        x = x + 1

    weight3= weight.reshape(F,HH*WW)
    input4=input4.reshape(10,676)
    input3=np.zeros((10,9,676)) #9,784
    output=np.zeros((10,8,676))
    print("da")
    for k in range(10-1):
        for i in range(9-1):
            for j in range(676-1):
              #  print(i,j,k,input2[0][k-1])
                input3[k][i][j]=input4[k][j]
 #   conv = FHEconv2dCQ(input,weight,bias)
    bias2 = np.zeros((8, 676))
    for i in range(8 - 1):
        for j in range(676 - 1):
            bias2[i][j] = bias[i]
  #  for i in range(10-1):
 #       output[i]=np.dot(weight3,input3[i])
  #  output= output.reshape((10,8,26,26))


    for i in range(10 - 1):
        #output[i] = np.dot(weight3, input3[i])
        output[i] = FHEDot(input3[i],weight3)
        output[i] = np.add(output[i].reshape(8, 676), bias2)
    output = output.reshape((10, 8, 26, 26))

    conv = torch.from_numpy(output).type(torch.cuda.FloatTensor)
    #print(weight)
    return conv



def conv2dCQ3(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()


    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape

    weight2= weight.reshape(1,-1)
    input2=input.reshape((10,1,28,28))
    input4=np.zeros(10*1*26*26)
    x = 0
    for k in range(10 - 1):
        for i in range(1 - 1):
            for j in range(28 - 1):
                for l in range(28 - 1):
                    if j == 0 or l == 0 or j == 27 or l == 27:
                        break
                    else:
                        input4[x] = input2[k][i][j][l]
                        x = x + 1

    weight3= weight.reshape(F,HH*WW)
    input4=input4.reshape(10,676)
    input3=np.zeros((10,9,676)) #9,784
    output=np.zeros((10,8,676))
    print("da")
    for k in range(10-1):
        for i in range(9-1):
            for j in range(676-1):
              #  print(i,j,k,input2[0][k-1])
                input3[k][i][j]=input4[k][j]
 #   conv = FHEconv2dCQ(input,weight,bias)
    bias2=np.zeros((8,676))
    for i in range(8-1):
        for j in range(676 - 1):
            bias2[i][j]=bias[i]
    for i in range(10-1):
        output[i]=np.dot(weight3,input3[i])
        output[i]=np.add(output[i].reshape(8,676),bias2)
    output= output.reshape((10,8,26,26))


    conv = torch.from_numpy(output).type(torch.cuda.FloatTensor)
    #print(weight)
    return conv


def conv2dCQ5(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()


    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape

    weight2= weight.reshape(1,-1)
    weight3= weight.reshape(F,HH*WW)
    input3=np.zeros((10,1,26*26,9)) #9,784
    output=np.zeros((10,8,26*26))
    print("da")
    x=-1
    for k in range(10-1):
        for i in range(1-1):
            for a in range(26-1):
                for b in range(26 - 1):
                    y=-1
                    for c in range(3 - 1):
                        for d in range(3 - 1):
                            x=x+1
                            y=y+1
                            input3[k][i][x][y]=input[k][i][a+c][b+d]

    bias2=np.zeros((8,676))
    for i in range(8-1):
        for j in range(676 - 1):
            bias2[i][j]=bias[i]


    for i in range(10 - 1):
        output[i]=np.dot(weight3,np.transpose(input3[i][0]))
        output[i] = np.add(output[i], bias2)
    print("ha")

    output = output.reshape((10, 8, 26, 26))

    conv = torch.from_numpy(output).type(torch.cuda.FloatTensor)
    #print(weight)
    return conv


def conv2dCQ6(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()


    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape

    weight2= weight.reshape(1,-1)
    weight3= np.zeros((8,9))
    input3=np.zeros((10,1,26*26,9)) #9,784
    output=np.zeros((10,8,26*26))
    print("da")
    x=-1
    for k in range(8-1):
        for i in range(1-1):
            y = -1
            for a in range(3-1):
                for b in range(3 - 1):
                    y=y+1
                    wight3[k][y]=wight[k][i][a][b]

    x=-1
    for k in range(10-1):
        for i in range(1-1):
            for a in range(26-1):
                for b in range(26 - 1):
                    y=-1
                    for c in range(3 - 1):
                        for d in range(3 - 1):
                            x=x+1
                            y=y+1
                            input3[k][i][x][y]=input[k][i][a+c][b+d]

    bias2=np.zeros((8,676))
    for i in range(8-1):
        for j in range(676 - 1):
            bias2[i][j]=bias[i]


    for i in range(10 - 1):
        output[i]=np.dot(weight3,np.transpose(input3[i][0]))
        for j in range(8-1):
            output[i][j]=np.add(output[i][j],bias[j])
     #   output[i] = np.add(output[i], np.transpose(bias))
    print("ha")

    output = output.reshape((10, 8, 26, 26))

    conv = torch.from_numpy(output).type(torch.cuda.FloatTensor)
   # print(weight)
    return conv
def conv2dCQ7(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()

    weight3= np.zeros((8,9))
    input3=np.zeros((64,1,26*26,9)) #9,784
    output=np.zeros((64,8,26*26))
    print("da")
    x=-1
    for k in range(8):
       # print('k=', k)
        for i in range(1):
         #   print('i=', i)
            y = -1
            for a in range(3):
                for b in range(3):
                    y=y+1
               #     print(weight[k][i][a][b])
                    weight3[k][y]=weight[k][i][a][b]

    x= -1
    for k in range(64):
        print('k=',k)
        for i in range(1):
            print('i=', i)
            x = -1
            for a in range(28-3+1):

                print('a=', a)
                for b in range(28-3+1):
                    print('b=', b)
                    y=-1
                    x = x + 1
                    for c in range(3):
                        for d in range(3):
                            y=y+1
                            print('x,y',x,y)
                            input3[k][i][x][3*c+d]=input[k][i][a+c][b+d]

    bias2=np.zeros((8,676))
    for i in range(8-1):
        for j in range(676 - 1):
            bias2[i][j]=bias[i]


    for i in range(64):
        output[i]=np.dot(weight3,np.transpose(input3[i][0]))
        for j in range(8):
            output[i][j]=np.add(output[i][j],bias[j])
     #   output[i] = np.add(output[i], np.transpose(bias))
    print("ha")

    output = output.reshape((64, 8, 26, 26))

    conv = torch.from_numpy(output).type(torch.cuda.FloatTensor)
#    print(weight)
    return conv

def conv2dCQ8(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()

    weight3= np.zeros((8,9))
    input3=np.zeros((2,1,26*26,9)) #9,784
    output=np.zeros((2,8,26*26))
    print("da")
    x=-1
    for k in range(8):
       # print('k=', k)
        for i in range(1):
         #   print('i=', i)
            y = -1
            for a in range(3):
                for b in range(3):
                    y=y+1
               #     print(weight[k][i][a][b])
                    weight3[k][y]=weight[k][i][a][b]

    x= -1
    for k in range(2):
       # print('k=',k)
        for i in range(1):
         #   print('i=', i)
            x = -1
            for a in range(28-3+1):

              #  print('a=', a)
                for b in range(28-3+1):
                   # print('b=', b)
                   # y=-1
                    x = x + 1
                    for c in range(3):
                        for d in range(3):
                        #    y=y+1
                            #print('x,y',x,y)
                            input3[k][i][x][3*c+d]=input[k][i][a+c][b+d]

    bias2=np.zeros((8,676))
    for i in range(8-1):
        for j in range(676):
            bias2[i][j]=bias[i]


    for i in range(2):
       # output[i]=np.dot(weight3,np.transpose(input3[i][0]))
        output[i] = FHEDot(weight3,input3[i][0])
        print(output[i])
        #for j in range(8):
        #    output[i][j]=np.add(output[i][j],bias[j])
        output[i] = np.add(output[i], bias2)
    print("ha")

    output = output.reshape((2, 8, 26, 26))

    conv = torch.from_numpy(output).type(torch.cuda.FloatTensor)
    print(weight)
    return conv


def conv2dCQ(input, weight, bias, stride=1,
                padding=0, dilation=1, groups=1):
#    encrypFHE4d(input)    #64,1,28,28
  #  encrypFHE4d(weight)   #8,1,3,3

    print("CONV2D_CQ")
    input = input.cpu().detach().numpy()
    weight = weight.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()


    N, C, H, W = input.shape
    F, _, HH, WW = weight.shape
    weight3= np.zeros((F,HH*WW))
    input3=np.zeros((N,C,(H-2)*(W-2),HH*WW)) #9,784
    output=np.zeros((N,F,(H-2)*(W-2)))
    print("da")
    x=-1
    for k in range(F):
       # print('k=', k)
        for i in range(C):
         #   print('i=', i)
            y = -1
            for a in range(HH):
                for b in range(WW):
                    y=y+1
               #     print(weight[k][i][a][b])
                    weight3[k][y]=weight[k][i][a][b]

    x= -1
    for k in range(N):
       # print('k=',k)
        for i in range(C):
         #   print('i=', i)
            x = -1
            for a in range(H-HH+1):

              #  print('a=', a)
                for b in range(W-WW+1):
                   # print('b=', b)
                   # y=-1
                    x = x + 1
                    for c in range(HH):
                        for d in range(WW):
                        #    y=y+1
                            #print('x,y',x,y)
                            input3[k][i][x][3*c+d]=input[k][i][a+c][b+d]

    bias2=np.zeros((F,(H-2)*(W-2)))
    for i in range(F):  #-1?
        for j in range((H-2)*(W-2)):
            bias2[i][j]=bias[i]


    for i in range(N):
       # output[i]=np.dot(weight3,np.transpose(input3[i][0]))
       output[i] = FHEDot(weight3, input3[i][0])
       output[i] = np.add(output[i], bias2)
     #   print(output[i])
        #for j in range(8):
        #    output[i][j]=np.add(output[i][j],bias[j])
    print("ha")

    output = output.reshape((N, F, H-2, W-2))

    conv = torch.from_numpy(output).type(torch.cuda.FloatTensor)

    print("DEBUG")
    return conv






def FHEConv2d(inputs_x, inputs_w, inputs_b):

    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    context=SEALContext.Create(parms)

    print_parameters(context)

    keygen = KeyGenerator(context)

    secret_key = keygen.secret_key()
    public_key = keygen.public_key()
    relin_keys = RelinKeys()
    gal_keys = GaloisKeys()

    if context.using_keyswitching():
        relin_keys = keygen.relin_keys()
        if not context.key_context_data().qualifiers().using_batching:
            print("Given encryption parameters do not support batching.")
            return 0
        gal_keys = keygen.galois_keys()

    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    ckks_encoder = CKKSEncoder(context)

    plain_x = Plaintext()
    plain_w = Plaintext()

    row_inputs_x = inputs_x.shape[0]  # 2
    col_inputs_x = inputs_x.shape[1]  # 3

    row_inputs_w = inputs_w.shape[0]  # 2
    col_inputs_w = inputs_w.shape[1]  # 3

    outputs_xw = np.zeros((row_inputs_x,row_inputs_w), dtype='float')
    plain = Plaintext()

    encrypted_100 = Ciphertext(context)
    encrypted_zeros = Ciphertext(context)

    inputs_100 = DoubleVector()
    zeros = DoubleVector()

    result1 = DoubleVector()

    for i in range(col_inputs_x):  #??????????????   2x2
        inputs_100.push_back(0)# 10000
        zeros.push_back(0)
    inputs_100[0]=1

    ckks_encoder.encode(inputs_100, scale, plain)#bianma
    encryptor.encrypt(plain, encrypted_100)
    ckks_encoder.encode(zeros, scale, plain)
    encryptor.encrypt(plain, encrypted_zeros)
    inputs_x_DoubleVector = DoubleVector()
    for i in range(col_inputs_x):
        inputs_x_DoubleVector.push_back(0)
    inputs_w_DoubleVector = DoubleVector()
    for i in range(col_inputs_w):
        inputs_w_DoubleVector.push_back(0)


    encrypted_x = Ciphertext(context)
    encrypted_w = Ciphertext(context)
    encrypted_xw = Ciphertext(context)
    add_mudi_Cipher = Ciphertext(context)
    result_Cipher = Ciphertext(context)
    mul_mudi_Cipher = Ciphertext(context)
    x=-1



    for i in range(row_inputs_x):
        for j in range(col_inputs_x):
            inputs_x_DoubleVector[j]=inputs_x[i][j]
        for k in range(row_inputs_w):
          #  inputs2 = DoubleVector()
            for l in range(col_inputs_w):
                inputs_w_DoubleVector[l]=inputs_w[k][l]
            ckks_encoder.encode(inputs_x_DoubleVector, scale, plain_x)
            ckks_encoder.encode(inputs_w_DoubleVector, scale, plain_w)
            encryptor.encrypt(plain_x, encrypted_x)
            encryptor.encrypt(plain_w, encrypted_w)
            evaluator.multiply(encrypted_x, encrypted_w, encrypted_xw)
            evaluator.relinearize_inplace(encrypted_xw, relin_keys)

            x = x+1
            for m  in range(col_inputs_x):
                if m == 0:
                    evaluator.multiply(encrypted_xw,encrypted_100,mul_mudi_Cipher)
                    evaluator.relinearize_inplace(mul_mudi_Cipher, relin_keys)
                    evaluator.multiply(encrypted_xw, encrypted_zeros, add_mudi_Cipher)
                    evaluator.relinearize_inplace( add_mudi_Cipher, relin_keys)
                    evaluator.add_inplace(add_mudi_Cipher, mul_mudi_Cipher)
                else:
                    evaluator.rotate_vector_inplace(encrypted_xw, 1, gal_keys)
                    evaluator.multiply(encrypted_xw, encrypted_100, mul_mudi_Cipher)
                    evaluator.relinearize_inplace(mul_mudi_Cipher, relin_keys)
                    evaluator.add_inplace(add_mudi_Cipher,mul_mudi_Cipher)
            if x ==0:
                evaluator.multiply(encrypted_xw, encrypted_zeros, result_Cipher)
                evaluator.relinearize_inplace(result_Cipher, relin_keys)
            else:
                evaluator.rotate_vector_inplace(add_mudi_Cipher, -int(x), gal_keys)
                evaluator.add_inplace(result_Cipher,add_mudi_Cipher)

  #  decryptor.decrypt(result_Cipher, plain)
   # ckks_encoder.decode(plain, result1)
   # print_vector(result1, 4, 7)
 #   return result_Cipher # cipher


    decryptor.decrypt(result_Cipher, plain)
    ckks_encoder.decode(plain, result1)
    #print_vector(result1, 4, 7)
    x=-1
    for i in range(row_inputs_x):
        for j in range(row_inputs_w):
            x =x+1
            outputs_xw[i][j] = float(result1[x])

    return outputs_xw





class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv2dCQ(_ConvNd):  #_ConvNd

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dCQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        print("!!!!!!!!!!!!!!!!")
        return conv2dCQ(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)

'''
class Conv2dCQ(torch.nn.Module):
    def __init__(self,n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        super(Conv2dCQ, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias',None)
        #self.reset_parameters()   # Default initializationb
    def forward(self, input):
        return LinearFunction(input,self.weight,self.bias) '''


