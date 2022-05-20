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
        self.conv1 = Conv2dCQ(1, 8, 3, 1) #Conv2dCQ(1, 8, 3, 1)  1*28*28->8*26*26,kernel=3*3,stri=1,no  add row
        self.conv2 = torch.nn.Conv2d(8, 12, 3, 1) # torch.nn.Conv2d(8, 12, 3, 1)
        self.dropout1 = Dropout2dCQ(0.25)
        self.dropout2 = Dropout2dCQ(0.5)
        self.fc1 = torch.nn.Linear(1728, 128)  #torch.nn.Linear(9216, 128)  1728
        #self.fc2 = torch.nn.Linear(128, 10)
        self.fc2 = torch.nn.Linear(128, 10)      #Linear(128, 10)   torch.nn.Linear(128, 10)


    def forward(self, x):

        x = self.conv1(x)
       # x = reluCQ(x)
        x=  F.relu(x)
        x = self.conv2(x)
#        x = max_pool2dCQ(x,2)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  #ok
        x = torch.flatten(x, 1)  #is ok if is encryt [] =vector
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x) #ok
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cpu or gpu
model = Net()  #模型
model.load_state_dict(torch.load("../../mnist_cnn_20200119.pt")) #加载模型，当时保存了参数
model.to(device)  #加载到设备中
model.eval()  #预测
print("Model:",model)
print("loading success")
#下面已注释掉的代码：保存模型参数到CNN_W_b_'+param_tensor+'.txt'中
'''
print("Model`s state_dict:")
for param_tensor in model.state_dict():
    #f = open(root + 'CNN_W_b_'+param_tensor+'.txt', 'w')
    #f.write(model.state_dict()[param_tensor].cpu().numpy())
    a=model.state_dict()[param_tensor].cpu().numpy().reshape(1,-1)
    #a=np.int64(a*100)
    np.savetxt('CNN_W_b_'+param_tensor+'.txt',a)
   # f.close()
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
'''

#加载test测试数据：已经实现分完
#train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())#,target_transform=transforms.Normalize((0.5,), (0.5,)))
test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())#,target_transform=transforms.Normalize((0.5,), (0.5,)))
#train_loader = DataLoader(dataset=train_data, batch_size=64*2, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2)

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
    out = model(batch_x) #结果
    #loss = loss_func(out, batch_y)
    # eval_loss += loss.data  #[0]
    # print(loss.data)
    i=i+1     #for循环的次数
    n=n+len(batch_y)
    pred = torch.max(out, 1)[1]
    num_correct = (pred == batch_y).sum()
    eval_acc += num_correct.data  #[0]
    if i == 1:  # batch*i= sum of test number
        break
    #....................end for

time_end_test = time.time()
time_sumepoch_test += (time_end_test - time_start_test) #test总时间秒
#print('Test Acc: {:.6f}'.format(eval_acc / (len(test_data))))

print("eval_acc=",eval_acc)
#print("len of batch_y=",len(batch_y))
print('TestForOne Acc: {:.6f}'.format(eval_acc / (n)))
print('time_sumepoch_test ',time_sumepoch_test )
print("THE end")