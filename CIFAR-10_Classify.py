import torch
import torchvision
import pickle
import os
import time
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary

class NET(nn.Module):
    def __init__(self):
        super().__init__()
        #in_shape3*32*32
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1)   #64*32*32
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)  #64*32*32
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)   #64*16*16
        self.bn1 = nn.BatchNorm2d(num_features=64)  #64*16*16
        self.relu1 = nn.ReLU()  #64*16*16

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1) #128*16*16
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1) #128*16*16
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)   #128*9*9
        self.bn2 = nn.BatchNorm2d(num_features=128) #128*9*9
        self.relu2 = nn.ReLU()  #128*9*9

        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)    #128*9*9
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)    #128*9*9
        self.conv7 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,padding=1)    #128*11*11
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1) #128*6*6
        self.bn3 = nn.BatchNorm2d(128)  #128*6*6
        self.relu3 = nn.ReLU()  #128*6*6

        self.conv8 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)    #256*6*6
        self.conv9 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)    #256*6*6
        self.conv10 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,padding=1)    #256*8*8
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1) #256*5*5
        self.bn4 = nn.BatchNorm2d(256)  #256*5*5
        self.relu4 = nn.ReLU()  #256*5*5

        self.conv11 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)   #512*5*5
        self.conv12 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)   #512*5*5
        self.conv13 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,padding=1)   #512*7*7
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1) #512*4*4
        self.bn5 = nn.BatchNorm2d(512)  #512*4*4
        self.relu5 = nn.ReLU()  #512*4*4

        self.fc14 = nn.Linear(512*4*4,1024) #1*1024
        self.drop1 = nn.Dropout2d() #1*1024
        self.fc15 = nn.Linear(1024,1024)    #1*1024
        self.drop2 = nn.Dropout2d() #1*1024
        self.fc16 = nn.Linear(1024,10)  #1*10

    def forward(self,x):
        x = x.to(device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x) 

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x) 

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
            
        #print("x shape:",x.size()) 
        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

class CIFAR10_Dataset(Data.Dataset):
    def __init__(self,train=True,transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        if self.train:
            self.train_data,self.train_labels = get_data(train)
            self.train_data = self.train_data.reshape((50000,3,32,32))
            self.train_data = self.train_data.transpose((0,2,3,1))
        else:
            self.test_data,self.test_labels = get_data()
            self.test_data = self.test_data.reshape((10000,3,32,32))
            self.test_data = self.test_data.transpose((0,2,3,1))
    
    def __getitem__(self, index):
        if self.train:
            img,label = self.train_data[index],self.train_labels[index]
        else:
            img,label = self.test_data[index],self.test_labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(label)
        
        return img,target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

transform = transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def Load_Data():

    trainset = CIFAR10(root='./data',train=True,download=True,transform=transform)
    trainloader = DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

    testset = CIFAR10(root='./data',train=False,download=True,transforms=transform)
    testloader = DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    
    return trainloader,testloader,classes


def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def get_data(train=False):
    data = None
    labels = None
    if train == True:
        for i in range(1,6):
            batch = unpickle('data/cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data,batch[b'data']])
            
            if i == 1:
                labels = batch[b'labels']
            else:
                labels = np.concatenate([labels,batch[b'labels']])
    else:
        batch = unpickle('data/cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    
    return data,labels

def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target



if __name__ == "__main__":
    CLASS_NUMs = 10
    BATCH_SIZE = 128
    EPOCH = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #trainloader,testloader,classes = Load_Data()
    train_data = CIFAR10_Dataset(True,transform,target_transform)
    print("size of train_data:{}".format(train_data.__len__()))
    test_data = CIFAR10_Dataset(False,transform,target_transform)
    print("size of test_data:{}".format(test_data.__len__()))
    train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

    net = NET()
    net.to(device)

    with torch.no_grad():
        for input_data,_ in train_loader:
            break
        summary(net,input_data.size()[1:])
    os.system("pause")

    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=5e-4)

    loss_fn = nn.CrossEntropyLoss()

    loss_list = []
    Accuracy = []

    for epoch in range(1,EPOCH+1):
        time_start = time.time()
        for step,(x,y) in enumerate(train_loader):
            #print(step,(x,y))
            b_x = Variable(x)
            b_y = Variable(y)
            output = net(b_x)
            b_x,b_y = b_x.to(device),b_y.to(device)
            loss = loss_fn(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                loss_list.append(loss)
        pre_correct = 0
        test_loader = Data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=True)
        for (x,y) in test_loader:
            b_x = Variable(x)
            b_y = Variable(y)
            b_x,b_y = b_x.to(device),b_y.to(device)
            output = net(b_x)
            pre = torch.max(output,1)[1]
            pre_correct = pre_correct + float(torch.sum(pre == b_y))
        print("EPOCH:{},ACC:{}%".format(epoch,(pre_correct/float(10000))*100))
        Accuracy.append((pre_correct/float(10000))*100)

        print("epoch %d cost %3f sec" % (epoch,time.time() - time_start))

    torch.save(net,'lenet_cifar_10.model')