import h5py
import tarfile
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
class X2Data(Dataset):
    def __init__(self):
        file_path = "backUp/data"
        symbol="\\"
        self.ouput = []
        self.inputs = []
        data_transform = transforms.Compose(transforms=[
            transforms.Resize((32,32)),

            transforms.ToTensor(),

            ])
        data_transform1 = transforms.Compose(transforms=[
            transforms.Resize((128,128)),

            transforms.ToTensor(),

        ])

        for root, dirs, files in os.walk(file_path):

            for file in files:
                img=Image.open(root + symbol + file)
                img=data_transform1(img)
                img=np.array(img,dtype="float")
                self.ouput.append(img)



        for root, dirs, files in os.walk(file_path):
            for file in files:
                img=Image.open(root + symbol + file)
                img = data_transform(img)
                img = np.array(img,dtype="float")
                self.inputs.append(img)
        self.len = len(self.ouput)
        self.ouput = np.array(self.ouput, dtype="float")
        self.inputs = np.array(self.inputs,dtype="float")


    def __getitem__(self, item):
        return torch.FloatTensor(self.inputs[item]),torch.FloatTensor(self.ouput[item])

    def __len__(self):
        return self.len



data = X2Data()
train_size = int(0.75*len(data))
test_size = len(data) - train_size


train_data,test_data = random_split(data,[train_size,test_size])

train_loader = DataLoader(train_data,batch_size=4,shuffle=True)
test_loader = DataLoader(test_data,batch_size=4,shuffle=True)

class Data(Dataset):
    def __init__(self):
        file_path = "data"
        symbol="\\"
        self.ouput = []
        self.inputs = []
        data_transform = transforms.Compose(transforms=[
            transforms.Resize((32,32)),

            transforms.ToTensor(),

            ])
        data_transform1 = transforms.Compose(transforms=[
            transforms.Resize((128,128)),

            transforms.ToTensor(),

        ])

        for root, dirs, files in os.walk(file_path):
            int = 0

            for file in files:
                int+=1
                if int<100 and int%10==0:
                    img=Image.open(root + symbol + file)
                    img=data_transform1(img)
                    img=np.array(img,dtype="float")
                    self.ouput.append(img)



        for root, dirs, files in os.walk(file_path):
            int = 0
            for file in files:
                int += 1
                if int < 100 and int % 10 == 0:
                    img=Image.open(root + symbol + file)
                    img = data_transform(img)
                    img = np.array(img,dtype="float")
                    self.inputs.append(img)
        self.len = len(self.ouput)
        self.ouput = np.array(self.ouput, dtype="float")
        self.inputs = np.array(self.inputs,dtype="float")


    def __getitem__(self, item):
        return torch.FloatTensor(self.inputs[item]),torch.FloatTensor(self.ouput[item])

    def __len__(self):
        return self.len
pdata=Data()
pdata = DataLoader(pdata,batch_size=4)




