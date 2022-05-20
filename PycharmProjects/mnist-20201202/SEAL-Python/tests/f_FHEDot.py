import torch
import torch.nn.functional as F
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

scale = pow(2.0, 40)
root=""

def FHEDot(inputs_x, inputs_w):
 #   global parms,poly_modulus_degree,context,keygen,secret_key,public_key,relin_keys,gal_keys
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
    print("    row_inputs_x = ",row_inputs_x)
    col_inputs_x = inputs_x.shape[1]  # 3
    print("    col_inputs_x = ",col_inputs_x)
    row_inputs_w = inputs_w.shape[0]  # 2
    print("    row_inputs_w = ",row_inputs_w)
    col_inputs_w = inputs_w.shape[1]  # 3
    print("    col_inputs_w = ",col_inputs_w)

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



    for i in range(row_inputs_x):#8,9
        for j in range(col_inputs_x):
            inputs_x_DoubleVector[j]=inputs_x[i][j]
       # x=-1
        for k in range(row_inputs_w):  #676
          #  inputs2 = DoubleVector()
            for l in range(col_inputs_w): #9
                inputs_w_DoubleVector[l]=inputs_w[k][l]
            ckks_encoder.encode(inputs_x_DoubleVector, scale, plain_x)
            ckks_encoder.encode(inputs_w_DoubleVector, scale, plain_w)
            encryptor.encrypt(plain_x, encrypted_x)
            encryptor.encrypt(plain_w, encrypted_w)
            evaluator.multiply(encrypted_x, encrypted_w, encrypted_xw)
            evaluator.relinearize_inplace(encrypted_xw, relin_keys)

            x = x+1
            #print("I,J,K,L,x=",i,j,k,l,x)
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
                evaluator.add_inplace(result_Cipher, add_mudi_Cipher)
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

def FHEDot3(inputs_x, inputs_w):
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    ckks_encoder = CKKSEncoder(context)
    plain_x = Plaintext()
    plain_w = Plaintext()

    row_inputs_x = inputs_x.shape[0]  # 2
    print("    row_inputs_x = ",row_inputs_x)
    col_inputs_x = inputs_x.shape[1]  # 3
    print("    col_inputs_x = ",col_inputs_x)
    row_inputs_w = inputs_w.shape[0]  # 2
    print("    row_inputs_w = ",row_inputs_w)
    col_inputs_w = inputs_w.shape[1]  # 3
    print("    col_inputs_w = ",col_inputs_w)

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



    for i in range(row_inputs_x):#8,9
        for j in range(col_inputs_x):
            inputs_x_DoubleVector[j]=inputs_x[i][j]
       # x=-1
        for k in range(row_inputs_w):  #676
          #  inputs2 = DoubleVector()
            for l in range(col_inputs_w): #9
                inputs_w_DoubleVector[l]=inputs_w[k][l]
            ckks_encoder.encode(inputs_x_DoubleVector, scale, plain_x)
            ckks_encoder.encode(inputs_w_DoubleVector, scale, plain_w)
            encryptor.encrypt(plain_x, encrypted_x)
            encryptor.encrypt(plain_w, encrypted_w)
            evaluator.multiply(encrypted_x, encrypted_w, encrypted_xw)
            evaluator.relinearize_inplace(encrypted_xw, relin_keys)

            x = x+1
            #print("I,J,K,L,x=",i,j,k,l,x)
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
                evaluator.add_inplace(result_Cipher, add_mudi_Cipher)
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


def FHEDot2(inputs_x, inputs_w):

    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 8192*4
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    context=SEALContext.Create(parms)

    print_parameters(context)

   # parms = context.first_context_data().parms()
  #  plain_modulus = parms.plain_modulus()
   # poly_modulus_degree = parms.poly_modulus_degree()

#    print("Generating secret/public keys: ", end="")
    keygen = KeyGenerator(context)
 #   print("Done")

    secret_key = keygen.secret_key()
    public_key = keygen.public_key()
    relin_keys = RelinKeys()
    gal_keys = GaloisKeys()

    if context.using_keyswitching():
       # print("Generating relinearization keys: ", end="")

        relin_keys = keygen.relin_keys()

        if not context.key_context_data().qualifiers().using_batching:
            print("Given encryption parameters do not support batching.")
            return 0

       # print("Generating Galois keys: ", end="")

        gal_keys = keygen.galois_keys()

    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    ckks_encoder = CKKSEncoder(context)
    result1 = DoubleVector()

    # How many times to run the test?
  #  count = 1

    # Populate a vector of floating-point values to batch.
    #   pod_vector = DoubleVector()
    #   slot_count =ckks_encoder.slot_count()
    #   print('\nslot_count=',slot_count)
    #   for i in range(5): #slot_count
    #      pod_vector.push_back(2.1 * float(1))

    #   print("Running tests ", end="")
    plain_x = Plaintext()
    plain_w = Plaintext()
   # inputs = DoubleVector()
  #  inputs2 = DoubleVector()

    #    inputs_x = np.array([[0,1.1,2],[3,4,5]])     #2*3=6
    row_inputs_x = inputs_x.shape[0]  # 2
    col_inputs_x = inputs_x.shape[1]  # 3
    # inputs_x = inputs_x.reshape(-1)
    #    inputs_w = np.array([[0,2,4],[1,3.1,5]])  #3*2=6

    row_inputs_w = inputs_w.shape[0]  # 2
    col_inputs_w = inputs_w.shape[1]  # 3

    outputs_xw = np.zeros(row_inputs_w * row_inputs_w, dtype='float')
    plain = Plaintext()
    plain1 = Plaintext()
    encrypted_1000 = Ciphertext(context)
    inputs_1000 = DoubleVector()
    x = 0
    for i in range(row_inputs_x*row_inputs_w):   #??????????????   2x2
        inputs_1000.push_back(0)# 10000
    inputs_1000[0]=1
  #  ckks_encoder.encode(inputs_1000, scale, plain)

   # print("inputs_1000=")
  #  print_vector(inputs_1000, 3, 7)
  #  inputs_1000[0]=0
 #   print(inputs_1000[0])
    ckks_encoder.encode(inputs_1000, scale, plain)
    encryptor.encrypt(plain, encrypted_1000)
    inputs_x_DoubleVector = DoubleVector()
    for i in range(col_inputs_x):
        inputs_x_DoubleVector.push_back(0)
    inputs_w_DoubleVector = DoubleVector()
    for i in range(col_inputs_w):
        inputs_w_DoubleVector.push_back(0)
   # inputs_w_DoubleVector=inputs_x.reshape(-1)

    encrypted_x = Ciphertext(context)
    encrypted_w = Ciphertext(context)
    encrypted_xw = Ciphertext(context)

    for i in range(row_inputs_x):
       # inputs = DoubleVector()

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
            evaluator.multiply(encrypted_w, encrypted_x, encrypted_w)

          #  decryptor.decrypt(encrypted_w, plain1)  # print encrypted1
         #   ckks_encoder.decode(plain1, result1)
         #   print("encrypted1=")
        #    print_vector(result1, 3, 7)

       #     print('i.j.k,l=', i, j, k, l)

            rotated_Plain = Plaintext()
            rotated_Cipher = Ciphertext()  # Plaintext()
            added_Cipher = Ciphertext()  # Plaintext()

    # rotated = encrypted_w#decryptor.decrypt(encrypted_w,rotated)
            # decryptor.decrypt(encrypted_w, rotated)
            decryptor.decrypt(encrypted_w, rotated_Plain)
            #rotated_Cipher = encrypted_w
            for m in range(col_inputs_x - 1):
               # print('m=', m)
                #rotated_Cipher = encrypted_w
                encryptor.encrypt(rotated_Plain, rotated_Cipher)
              #  ckks_encoder.decode(rotated_Plain, result1)
            #    print("!!in_m _rotated_Plain")
           #     print_vector(result1, 3, 7)
                # rotated = encrypted_w

                evaluator.relinearize_inplace(encrypted_w, relin_keys)

                # evaluator.relinearize_inplace(rotated2, relin_keys)
                evaluator.rotate_vector_inplace(encrypted_w, 1, gal_keys)

                decryptor.decrypt(encrypted_w, plain1)  # print encrypted1
                ckks_encoder.decode(plain1, result1)
                print("!in_m _rotate1_encrypted_w")
                print_vector(result1, 3, 7)
                # evaluator.add_inplace(encrypted_w, evaluator.rotate_vector_inplace(rotated,1,gal_keys))
                #   evaluator.relinearize_inplace(rotated2, relin_keys)
               #
                evaluator.add(rotated_Cipher, encrypted_w, rotated_Cipher)
                # evaluator.multiply_inplace(rotated2, encrypted_1000)

                # evaluator.relinearize_inplace(rotated2, relin_keys)
                decryptor.decrypt(rotated_Cipher,rotated_Plain)  # print encrypted1
                ckks_encoder.decode(rotated_Plain, result1)
                print("!in_m_after_add_rencrypted_w")
                print_vector(result1, 3, 7)

                evaluator.multiply_inplace(rotated_Cipher, encrypted_1000)

                decryptor.decrypt(rotated_Cipher, plain)  # print encrypted1
                ckks_encoder.decode(plain, result1)
                print("!in_m_after_add_rencrypted_w")
                print_vector(result1, 3, 7)

            x = x + 1
            if x == 1:
                add_berfor = Ciphertext(context)
                add_berfor = rotated_Cipher
            if x != 1:
                evaluator.relinearize_inplace(rotated_Cipher, relin_keys)
                evaluator.rotate_vector_inplace(rotated_Cipher, -int(x - 1), gal_keys)
                evaluator.add_inplace(add_berfor, rotated_Cipher)
    decryptor.decrypt(add_berfor, plain)
    result1 = DoubleVector()  # print
    ckks_encoder.decode(plain, result1)

    for i in range(row_inputs_w * row_inputs_x):
        outputs_xw[i] = float(result1[i])

    outputs_xw = outputs_xw.reshape(row_inputs_x, row_inputs_w)
    print("!!result1-=")
    print_vector(result1, 18, 7)

    print(" Done\n", flush=True)
    return outputs_xw



#inputs_x = np.array([[0,1,2,5],[3,4,5,5]])

##inputs_w = np.array([[0,2,5,4],[1,3,5,6],[3,5,8,3.3]])  #3*2=6
##outputs_xw = FHEDot(inputs_x,inputs_w)
#print(outputs_xw)
