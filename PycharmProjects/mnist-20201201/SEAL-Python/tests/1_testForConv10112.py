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
from f_LinearCQ import *   #f_**  is  all already translat by CQ
from f_Conv2dCQ import *
from f_dropoutCQ import *
from CQ_Linear import *
from CQ_Conv2d import *
from CQ_Relu import *
from CQ_MaxPool import *
from CQ_Sigmoid import *
import time
import math
import random
from seal import *
from seal_helper import *
from importlib import reload
#reload(f_LinearCQ)
#reload(f_Conv2dCQ)
#reload(f_dropoutCQ)
#reload(f_functionalCQ)
reload(torch)
root="../../"

def default_loader(path):
    return Image.open(root+path).convert('L') # gray:L or coler:RGB
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



def SigmoidFunction(input):
   # print("SigmoidFunction")
    input = input.cpu().detach().numpy()
  #  print("SigmoidFunction:")
    A,B,H,W = input.shape
    #output=np.zeros((A.B.H.W))
    for i in range(A):
        for j in range(B):
            for a in range(H):
                for b in range(W):
                    #input[i][j][a][b]=1.0/(1+np.exp(-(float)(input[i][j][a][b])))
                    input[i][j][a][b] = 1.0 / (1 + 1+(-input[i][j][a][b])+np.power(-input[i][j][a][b],2)/2+np.power(-input[i][j][a][b],3)/6+np.power(-input[i][j][a][b],4)/24)
                    #input[i][j][a][b] = 1/2+1/4*input[i][j][a][b]\
                    #                    -1/48*np.power(input[i][j][a][b],3)\
                   #                     +1/480*np.power(input[i][j][a][b],5)\
                   #                     -17/86040*np.power(input[i][j][a][b],7)\
                    #                    +31/1451520*np.power(input[i][j][a][b],9)
    # output=1.0/(1+np.exp(-(float)(input)))
    input = torch.from_numpy(input).type(torch.cuda.FloatTensor)
    return input

def EncryptFunction(input):

    return input

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2dCQ(1, 6, 3, 1) #1Conv2dCQ(1, 6, 3, 1)*28*28->32*28*28,kernel=3*3,stri=1,no  add row
        self.conv2 = torch.nn.Conv2d(6, 7, 3, 1)
       # self.dropout1 = torch.nn.Dropout2d(0.25) 后面暂时没用
      #  self.dropout2 = torch.nn.Dropout2d(0.5)  后面暂时没用 先注释掉 后面再加
        self.fc1 = torch.nn.Linear(4032, 128)
        #self.fc2 = torch.nn.Linear(128, 10)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
       # x = EncryptFunction(x) #先对数据进行加密
        x = self.conv1(x)
        x = SigmoidFunction(x)
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
model.load_state_dict(torch.load("../../mnist_cnn_20200122.pt"))
model.to(device)
model.eval()
print("Model:",model)
print("loading success")
print("Model`s state_dict:")
for param_tensor in model.state_dict():
    #f = open(root + 'CNN_W_b_'+param_tensor+'.txt', 'w')
    #f.write(model.state_dict()[param_tensor].cpu().numpy())
    a=model.state_dict()[param_tensor].cpu().numpy().reshape(1,-1)
    #a=np.int64(a*100)

    np.savetxt('CNN_W_b_'+param_tensor+'.txt',a)
   # f.close()
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())

#train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())#,target_transform=transforms.Normalize((0.5,), (0.5,)))
test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())#,target_transform=transforms.Normalize((0.5,), (0.5,)))
#train_loader = DataLoader(dataset=train_data, batch_size=64*2, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=25)

#loss_func = torch.nn.CrossEntropyLoss()

time_sumepoch_test=0
#eval_loss = 0.
eval_acc = 0.
time_start_test = time.time()
i=0
n=0





for batch_x, batch_y in test_loader:
#/...................star for
    batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)

    #  batch_x=(batch_x*255).type(torch.cuda.IntTensor)  #first *255,then to INT,to Float
    # batch_x = batch_x.type(torch.cuda.FloatTensor)

    #out = model(batch_x.trunc())
  #  print("!")
    out = model(batch_x)
    #loss = loss_func(out, batch_y)
    # eval_loss += loss.data  #[0]
    # print(loss.data)
    i=i+1     #number of cycles of for
    n=n+len(batch_y)
    pred = torch.max(out, 1)[1]

    num_correct = (pred == batch_y).sum()

    eval_acc += num_correct.data  #[0]
    if i == 1:  # batch*i= sum of test number
        break
    #....................end for

time_end_test = time.time()
time_sumepoch_test += (time_end_test - time_start_test)
#print('Test Acc: {:.6f}'.format(eval_acc / (len(test_data))))

print("eval_acc=",eval_acc)
#print("len of batch_y=",len(batch_y))
print('TestForOne Acc: {:.6f}'.format(eval_acc / (n)))

print('time_sumepoch_test ',time_sumepoch_test )

print("THE end")