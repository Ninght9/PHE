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
#np.set_printoptions(linewidth=50)

def rand_int():
    return int(random.random()*(10**10)) #10^10

def bfv_performance_test(context):
    print_parameters(context)

    parms = context.first_context_data().parms()
    plain_modulus = parms.plain_modulus()
    poly_modulus_degree = parms.poly_modulus_degree()

    print("Generating secret/public keys: ", end="")
    keygen = KeyGenerator(context)
    print("Done")

    secret_key = keygen.secret_key()
    public_key = keygen.public_key()
    relin_keys = RelinKeys()
    gal_keys = GaloisKeys()

    if context.using_keyswitching():
        # Generate relinearization keys.
        print("Generating relinearization keys: ", end="")

        relin_keys = keygen.relin_keys()


        if not context.key_context_data().qualifiers().using_batching:
            print("Given encryption parameters do not support batching.")
            return 0

        print("Generating Galois keys: ", end="")

        gal_keys = keygen.galois_keys()


    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    batch_encoder = BatchEncoder(context)
    encoder = IntegerEncoder(context)



    # How many times to run the test?
    count = 1

    # Populate a vector of values to batch.
    slot_count = batch_encoder.slot_count()
    pod_vector = uIntVector()

    for i in range(slot_count):# 4096 .....
        pod_vector.push_back(int(i)% plain_modulus.value())  # random choose 10**10
    print("Running tests ", end="")

    for i in range(count):
        '''
        [Batching]
        There is nothing unusual here. We batch our random plaintext matrix
        into the polynomial. Note how the plaintext we create is of the exactly
        right size so unnecessary reallocations are avoided.
        '''
        plain = Plaintext(parms.poly_modulus_degree(), 0)

        batch_encoder.encode(pod_vector, plain)  #pod_vector <-plain


        '''
        [Unbatching]
        We unbatch what we just batched.
        '''
        pod_vector2 = uIntVector()

        batch_encoder.decode(plain, pod_vector2)

        for j in range(slot_count):
            #print(pod_vector[j], pod_vector2[j])
            if pod_vector[j] != pod_vector2[j]:  #the same vector == vector2

                raise Exception("Batch/unbatch failed. Something is wrong.")



        '''
        [Encryption]
        We make sure our ciphertext is already allocated and large enough
        to hold the encryption with these encryption parameters. We encrypt
        our random batched matrix here.
        '''
        encrypted = Ciphertext()

        encryptor.encrypt(plain, encrypted)


        '''
        [Decryption]
        We decrypt what we just encrypted.
        '''
        plain2 = Plaintext(poly_modulus_degree, 0)

        decryptor.decrypt(encrypted, plain2)  #encrypted->plain2

        if plain.to_string() != plain2.to_string():   #the same
            raise Exception("Encrypt/decrypt failed. Something is wrong.")

        '''
        [Add]
        We create two ciphertexts and perform a few additions with them.
        '''
        plain3 = Plaintext(poly_modulus_degree, 0)
        plain4 = Plaintext(poly_modulus_degree, 0)



        encrypted1 = Ciphertext()
        encryptor.encrypt(encoder.encode(11), encrypted1)  #11

        decryptor.decrypt(encrypted1, plain3)  # encrypted->plain3    #    decryptor.decrypt(encrypted2, plain4)  # encrypted->plain4
        print('plain3=', plain3.to_string())


        encrypted2 = Ciphertext(context)
        encryptor.encrypt(encoder.encode(12), encrypted2)  #12



        decryptor.decrypt(encrypted2, plain4)  # encrypted2->plain4 12
        print('plain4=', plain4.to_string())

        evaluator.add_inplace(encrypted1, encrypted1) # encrypted1<=encrypted1+encrypted1 22

        decryptor.decrypt(encrypted1, plain3)  # encrypted1->plain3 22
        decryptor.decrypt(encrypted2, plain4)  # encrypted2->plain4 12
        print('!plain3=', plain3.to_string())
        print('plain4=', plain4.to_string())

        evaluator.add_inplace(encrypted2, encrypted2)  #12+12=24
       # print("encrypted1",encrypted1.to_string())
        decryptor.decrypt(encrypted2, plain4)  # encrypted->plain4
        print('plain4=', plain4.to_string())

        evaluator.add_inplace(encrypted1, encrypted2) #46=22+24

        decryptor.decrypt(encrypted1, plain3)  # encrypted->plain3  46
        decryptor.decrypt(encrypted2, plain4)  # encrypted->plain4  24
        print('plain3=',plain3.to_string())
        print('plain4=', plain4.to_string())

       # encrypted1.reserve(3)

      #  decryptor.decrypt(encrypted1, plain3)  # encrypted->plain3
        #print('plain3=',plain3.to_string())

        evaluator.multiply_inplace(encrypted1, encrypted2)
        decryptor.decrypt(encrypted1, plain3)  # encrypted->plain3 1104
        decryptor.decrypt(encrypted2, plain4)  # encrypted->plain4 24
        print(encoder.encode(1104).to_string())
        print(plain3.to_string())
        #if plain4.to_string() == string(encoder.encode(1104)):
         #   print("the same")
        print('!!!!plain3=', plain3.to_string())
        print('!!!!plain4=', plain4.to_string())



       # encoder.decode(plain4)
        # Print a dot to indicate progress.
        print(".", end="", flush=True)
    print(" Done", flush=True)

def ckks_performance_test(context,inputs_x,inputs_w):
    print_parameters(context)

    parms = context.first_context_data().parms()
    plain_modulus = parms.plain_modulus()
    poly_modulus_degree = parms.poly_modulus_degree()

    print("Generating secret/public keys: ", end="")
    keygen = KeyGenerator(context)
    print("Done")

    secret_key = keygen.secret_key()
    public_key = keygen.public_key()
    relin_keys = RelinKeys()
    gal_keys = GaloisKeys()

    if context.using_keyswitching():
        print("Generating relinearization keys: ", end="")

        relin_keys = keygen.relin_keys()



        if not context.key_context_data().qualifiers().using_batching:
            print("Given encryption parameters do not support batching.")
            return 0

        print("Generating Galois keys: ", end="")

        gal_keys = keygen.galois_keys()


    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    ckks_encoder = CKKSEncoder(context)
    result1 = DoubleVector()


    # How many times to run the test?
    count = 1

    # Populate a vector of floating-point values to batch.
 #   pod_vector = DoubleVector()
 #   slot_count =ckks_encoder.slot_count()
 #   print('\nslot_count=',slot_count)
#   for i in range(5): #slot_count
#      pod_vector.push_back(2.1 * float(1))

 #   print("Running tests ", end="")
    plain_x= Plaintext()
    plain_w= Plaintext()
    inputs = DoubleVector()
    inputs2 = DoubleVector()

#    inputs_x = np.array([[0,1.1,2],[3,4,5]])     #2*3=6
    row_inputs_x=inputs_x.shape[0]#2
    col_inputs_x=inputs_x.shape[1]#3
   # inputs_x = inputs_x.reshape(-1)
#    inputs_w = np.array([[0,2,4],[1,3.1,5]])  #3*2=6

    row_inputs_w=inputs_w.shape[0]#2
    col_inputs_w=inputs_w.shape[1]#3

    outputs_xw = np.zeros(row_inputs_w*row_inputs_w,dtype='float')
    plain = Plaintext()
    plain1 = Plaintext()
    encrypted_1000 = Ciphertext(context)
    inputs_1000 = DoubleVector()
    x = 0
    for i in range(row_inputs_x):
        if i==0:
            inputs_1000.push_back(1)
        inputs_1000.push_back(0)    #10000
    ckks_encoder.encode(inputs_1000,scale,plain)

    print("inputs_1000=")
    print_vector(inputs_1000, 3, 7)

    encryptor.encrypt(plain,encrypted_1000)
    for i in range(row_inputs_x):
        inputs = DoubleVector()

        for j in range(col_inputs_x):
            inputs.push_back(inputs_x[i][j])

        for k in range(row_inputs_w):
            inputs2 = DoubleVector()
            for l in range(col_inputs_w):
                inputs2.push_back(inputs_w[k][l])
            encrypted_x = Ciphertext(context)
            encrypted_w = Ciphertext(context)
            ckks_encoder.encode(inputs, scale, plain_x)
            ckks_encoder.encode(inputs2, scale, plain_w)
            encryptor.encrypt(plain_x, encrypted_x)
            encryptor.encrypt(plain_w, encrypted_w)
            evaluator.multiply_inplace(encrypted_w,encrypted_x)

            decryptor.decrypt(encrypted_w, plain1)  # print encrypted1
            ckks_encoder.decode(plain1, result1)
            print("encrypted1=")
            print_vector(result1, 3, 7)


            print('i.j.k,l=',i,j,k,l)
            rotated = Plaintext()
            rotated2 = Ciphertext()#Plaintext()
            rotated3 = Ciphertext()
            rotated4 = Ciphertext()
           # rotated = encrypted_w#decryptor.decrypt(encrypted_w,rotated)
           # decryptor.decrypt(encrypted_w, rotated)
            decryptor.decrypt(encrypted_w, rotated)

            for m in range(col_inputs_x-1):
                print('m=',m)



                encryptor.encrypt(rotated,rotated2)
                ckks_encoder.decode(rotated, result1)
                print("!!in_m _rotated")
                print_vector(result1, 3, 7)
               # rotated = encrypted_w

                evaluator.relinearize_inplace(encrypted_w,relin_keys)


               # evaluator.relinearize_inplace(rotated2, relin_keys)
                evaluator.rotate_vector_inplace(encrypted_w,1,gal_keys)

                decryptor.decrypt(encrypted_w, plain1)  # print encrypted1
                ckks_encoder.decode(plain1, result1)
                print("!in_m _rotate1_encrypted_w")
                print_vector(result1, 3, 7)
               # evaluator.add_inplace(encrypted_w, evaluator.rotate_vector_inplace(rotated,1,gal_keys))
             #   evaluator.relinearize_inplace(rotated2, relin_keys)
                evaluator.add_inplace(rotated2,encrypted_w)
                #evaluator.multiply_inplace(rotated2, encrypted_1000)

                #evaluator.relinearize_inplace(rotated2, relin_keys)
                decryptor.decrypt(rotated2, rotated)  # print encrypted1
                ckks_encoder.decode(rotated, result1)
                print("!in_m_after_add_rencrypted_w")
                print_vector(result1, 3, 7)

                evaluator.multiply_inplace(rotated2, encrypted_1000)

                decryptor.decrypt(rotated2, plain)  # print encrypted1
                ckks_encoder.decode(plain, result1)
                print("!in_m_after_add_rencrypted_w")
                print_vector(result1, 3, 7)

            x =x+1
            if x==1:

                add_berfor = Ciphertext(context)
                add_berfor = rotated2
            if x!=1:
                evaluator.relinearize_inplace(rotated2, relin_keys)
                evaluator.rotate_vector_inplace(rotated2, -int(x-1), gal_keys)
                evaluator.add_inplace(add_berfor,rotated2)
    decryptor.decrypt(add_berfor,plain)
    result1 = DoubleVector()  # print
    ckks_encoder.decode(plain, result1)


    for i in range(row_inputs_w*row_inputs_x):
        outputs_xw[i]=float(result1[i])

    outputs_xw=outputs_xw.reshape(row_inputs_x,row_inputs_w)
    print("!!result1-=")
    print_vector(result1, 18, 7)


    print(" Done\n", flush=True)

    """

 #   inputs_w = inputs_w.reshape(-1)
#    print(inputs_x[10])
#    print(inputs_w[16])
   # print(inputs_w.shape)
   # print(inputs_w.shape[0]) #3
 #   row_w=inputs_w.shape[0]
  #  col_w=inputs_w.shape[1]
   # print(inputs_w.shape[1])#10
    for i in range(6):
        inputs.push_back(inputs_x[i])
    for i in range(6):
        inputs2.push_back(inputs_w[i])

    encrypted_x = Ciphertext(context)
    encrypted_w = Ciphertext(context)
    ckks_encoder.encode(inputs, scale, plain_x)
    ckks_encoder.encode(inputs2,scale,plain_w)
    encryptor.encrypt(plain_x, encrypted_x)
    encryptor.encrypt(plain_w, encrypted_w)

    #for i in range(log2(poly_modulus_degree))
    rotated = Ciphertext(context)
    rotated = encrypted_w
    print('!')
    print(int(math.log2(poly_modulus_degree)))

   # for i in range(5):  #15  int(math.log2(poly_modulus_degree))
     #   encryptor.encrypt(plain_w, encrypted_w)

     #   evaluator.rotate_vector_inplace(encrypted_w,pow(2,i),gal_keys)
     #   evaluator.add_inplace(rotated,encrypted_w)

   # evaluator.multiply_inplace(encrypted_x,encrypted_w)


    decryptor.decrypt(rotated,plain_w)
    result_w = DoubleVector()  # print
    ckks_encoder.decode(plain_w, result_w)
    #print(int(result1[1]))
    print("encrypted=")
    print_vector(result_w, 12, 7)
 #   print_matrix(result_w, 3)
    for i in range(count):
        '''
        [Encoding]
        For scale we use the square root of the last coeff_modulus prime
        from parms.
        
        '''
 #       plain = Plaintext()
        plain = Plaintext(parms.poly_modulus_degree() *
                          len(parms.coeff_modulus()), 0) #4096*2
       # print('parms.poly_modulus_degree()',parms.poly_modulus_degree())
       # print(len(parms.coeff_modulus()))
        # [Encoding]
#        scale = math.sqrt(parms.coeff_modulus()[-1].value()) #262143.53125148825
      #  print('scale:',scale)



#        ckks_encoder.encode(pod_vector, scale, plain)  #pod_vector->plain
    #    print(plain)

        #[Decoding]
#        pod_vector2 = DoubleVector()

 #       ckks_encoder.decode(plain, pod_vector2)  #pod_vector->plain


        # [Encryption]
 #       encrypted = Ciphertext(context)

 #       encryptor.encrypt(plain, encrypted)


        # [Decryption]
        plain2 = Plaintext(poly_modulus_degree, 0)

  #      decryptor.decrypt(encrypted, plain2)


        # [Add]
        plain_result1 = Plaintext(poly_modulus_degree, 0)
        plain_result2 = Plaintext(poly_modulus_degree, 0)  #(poly_modulus_degree, 0)

        encrypted1 = Ciphertext(context)


        ckks_encoder.encode(2.1,scale, plain)
        result1 = DoubleVector()                    #print
        ckks_encoder.decode(plain,result1)
        print("plain=")
        print_vector(result1, 3, 7)

        encryptor.encrypt(plain, encrypted1)
        encrypted2 = Ciphertext(context)
        ckks_encoder.encode(3.2,scale, plain2)
        encryptor.encrypt(plain2, encrypted2)


        result2 = DoubleVector()     #print
        ckks_encoder.decode(plain2, result2)
        print("plain2=")
        print_vector(result2, 3, 7)

        decryptor.decrypt(encrypted1, plain_result1)  #print encrypted1
        ckks_encoder.decode(plain_result1, result1)
        print("encrypted1=")
        print_vector(result1, 3, 7)

        evaluator.add_inplace(encrypted1, encrypted1)

        decryptor.decrypt(encrypted1, plain_result1)  #print platin
        decryptor.decrypt(encrypted2, plain_result2)
        ckks_encoder.decode(plain_result1, result1)
        ckks_encoder.decode(plain_result2, result2)
        print("encrypted1=")
        print_vector(result1, 3, 7)
        print("encrypted2=")
        print_vector(result2, 3, 7)

        evaluator.add_inplace(encrypted2, encrypted2)
        evaluator.add_inplace(encrypted1, encrypted2)

        decryptor.decrypt(encrypted1, plain_result1)  #print platin
        decryptor.decrypt(encrypted2, plain_result2)
        ckks_encoder.decode(plain_result1, result1)
        ckks_encoder.decode(plain_result2, result2)
        print("encrypted1=")
        print_vector(result1, 3, 7)
        print("encrypted2=")
        print_vector(result2, 3, 7)  #en1 =10.6, en2=6.4

        # [Multiply]
#        encrypted1.reserve(3)

   #     evaluator.rescale_to_next_inplace(encrypted1)
    #    evaluator.rescale_to_next_inplace(encrypted2)
#        evaluator.multiply_inplace(encrypted1, encrypted2)


        # [Multiply Plain]
        print("!")
        plain3 = Plaintext(poly_modulus_degree, 0)
        ckks_encoder.encode(2,scale,plain3)
        #evaluator.rescale_to_next_inplace(plain3)
        evaluator.multiply_plain_inplace(encrypted2, plain3)



        decryptor.decrypt(encrypted1, plain_result1)  # print platin
        decryptor.decrypt(encrypted2, plain_result2)
        ckks_encoder.decode(plain_result1, result1)
        ckks_encoder.decode(plain_result2, result2)
        print("!!encrypted1=")
        print_vector(result1, 3, 7)
        print("!!encrypted2=")
        print_vector(result2, 3, 7)  # en1 =10.6, en2=6.4
        # [Square]

        evaluator.square_inplace(encrypted2)


        if context.using_keyswitching():

            # [Relinearize]

            evaluator.relinearize_inplace(encrypted1, relin_keys)


            # [Rescale]

            evaluator.rescale_to_next_inplace(encrypted1)


            # [Rotate Vector]

            evaluator.rotate_vector_inplace(encrypted, 1, gal_keys)
            evaluator.rotate_vector_inplace(encrypted, -1, gal_keys)


            # [Rotate Vector Random]
            random_rotation = int(rand_int() % ckks_encoder.slot_count())

            evaluator.rotate_vector_inplace(
                encrypted, random_rotation, gal_keys)


            # [Complex Conjugate]

            evaluator.complex_conjugate_inplace(encrypted, gal_keys)
"""



def example_ckks_performance_default(inputs_x,inputs_w):
    print_example_banner(
        "CKKS Performance Test with Degrees: 4096, 8192, and 16384")

    parms = EncryptionParameters(scheme_type.CKKS)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    ckks_performance_test(SEALContext.Create(parms),inputs_x,inputs_w)




def example_bfv_performance_default():
    print_example_banner(
        "BFV Performance Test with Degrees: 4096")

    parms = EncryptionParameters(scheme_type.BFV)
    poly_modulus_degree = 4096
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    parms.set_plain_modulus(786433)
    bfv_performance_test(SEALContext.Create(parms))

#example_bfv_performance_default()
#inputs_x = np.array([[0,1.1,2],[3,4,5]])
#inputs_w = np.array([[0,2,4],[1,3.1,5]])  #3*2=6
#example_ckks_performance_default(inputs_x,inputs_w)


# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('L') # gray:L
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
           # print(words)
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        #print(img)
        if self.transform is not None:
            img = self.transform(img)   #/255   (BGR)->(c,H,W)
            #print(img.numpy)
            #img = self.target_transform(img)
            #print(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())#,target_transform=transforms.Normalize((0.5,), (0.5,)))
test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())#,target_transform=transforms.Normalize((0.5,), (0.5,)))
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)


#-----------------create the Net and training------------------------

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, 1) #1*28*28->32*28*28,kernel=3*3,stri=1,no  add row
        self.conv2 = torch.nn.Conv2d(6, 7, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(4032, 128)
        #self.fc2 = torch.nn.Linear(128, 10)
        self.fc2 = torch.nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)

        x = self.conv2(x)
   #     x = F.max_pool2d(x, 2)
      #  x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
      #  x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #"cpu"

model = Net()
#if torch.cuda.device_count() >1 :
 #   print("Use",torch.cuda.device_count(),"GPUS")
 #   model = torch.nn.DataParallel(model,device_ids=[0,1])
model.to(device)
print(model)
'''print("Model`s state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())'''

optimizer = torch.optim.Adam(model.parameters())

'''print("Optimizer's stata_dict")
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])'''


loss_func = torch.nn.CrossEntropyLoss()

time_sumepoch_train=0
time_sumepoch_test=0

for epoch in range(20*3):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------

    train_loss = 0.
    train_acc = 0.
    time_start_train = time.time()
    for batch_x, batch_y in train_loader:
        #batch_x_numpy = batch_x.numpy()*255
        #print(batch_x_numpy)
        #batch_y_numpy = batch_y.numpy()
        #print(batch_y_numpy)
        batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data   #[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data #[0]
        optimizer.zero_grad()   #
        loss.backward()          #
        optimizer.step()        #
    time_end_train = time.time()
    time_sumepoch_train+=(time_end_train-time_start_train)
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    time_start_test = time.time()
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)
        #print(int(batch_x*255))
        out = model(batch_x)
        #print("out={},batch_y={}".format(out,batch_y))
        loss = loss_func(out, batch_y)
        eval_loss += loss.data  #[0]
       # print(loss.data)
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data  #[0]
    time_end_test = time.time()
    time_sumepoch_test += (time_end_test - time_start_test)
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))


print("Sum test time of epoch{}={},average of each epoch test time={}".format(int(epoch+1),time_sumepoch_test,time_sumepoch_test/(epoch+1)))
print("Sum train time of epoch{}={},average of each epoch train time={}".format(int(epoch+1),time_sumepoch_train,time_sumepoch_train/(epoch+1)))

torch.save(model.state_dict(), "mnist_cnn_20200122.pt")