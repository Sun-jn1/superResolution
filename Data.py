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
        file_path = "benchmark"
        # documents=["Set14"]
        documents=["B100","Set5","Urban100"]
        symbol="\\"
        TruthValue = "HR"
        X2Resolution = "LR_bicubic\\X4"
        self.ouput = []
        self.inputs = []
        data_transform = transforms.Compose(transforms=[
            transforms.Resize((180,180)),

            transforms.ToTensor(),
            # transforms.Normalize([0.42477179, 0.43378679, 0.3698565], [0.19699529, 0.1823555, 0.1743905]),
            ])
        for document in documents:
            for root, dirs, files in os.walk(file_path+symbol+document+symbol+TruthValue):

                for file in files:
                    img=Image.open(root + symbol + file)
                    img=data_transform(img)
                    img=np.array(img,dtype="float")
                    self.ouput.append(img)



                    # print(img.shape)

            for root, dirs, files in os.walk(file_path + symbol + document + symbol + X2Resolution):
                for file in files:
                    img=Image.open(root + symbol + file)
                    img = data_transform(img)
                    img = np.array(img,dtype="float")
                    self.inputs.append(img)
        self.len = len(self.ouput)
        # self.len = 15

        # self.ouput = np.array(self.ouput)
        self.ouput = np.array(self.ouput, dtype="float")

        # for index in range(self.len):
        #     img = self.inputs[index]
        #     for i in range(3):
        #         self.mean[i]+=img[i,:,:].mean()
        #         self.std[i]+=img[i,:,:].std()
        #
        # self.mean=np.array(self.mean)
        # self.std=np.array(self.std)
        # self.mean = self.mean/self.len
        # self.std = self.std/self.len

        # self.inputs = np.array(self.inputs)
        self.inputs = np.array(self.inputs,dtype="float")


    def __getitem__(self, item):
        return torch.FloatTensor(self.inputs[item]),torch.FloatTensor(self.ouput[item])

    def __len__(self):
        return self.len



data = X2Data()
train_size = int(0.75*len(data))
test_size = len(data) - train_size


train_data,test_data = random_split(data,[train_size,test_size])
# train_loader = DataLoader(train_data,batch_size=1,shuffle=False)
# test_loader = DataLoader(test_data,batch_size=1,shuffle=False)
train_loader = DataLoader(train_data,batch_size=4,shuffle=True)
test_loader = DataLoader(test_data,batch_size=4,shuffle=True)
# class X3Data(Dataset):
#     def __init__(self):
#         file_path = "benchmark"
#         documents = ["B100", "Set5", "Set14", "Urban100"]
#         symbol = "\\"
#         TruthValue = "HR"
#         X3Resolution = "LR_bicubic\\X3"
#         self.ouput = []
#         self.inputs = []
#         for document in documents:
#             for root, dirs, files in os.walk(file_path + symbol + document + symbol + TruthValue):
#                 self.ouput = self.ouput + files
#             for root, dirs, files in os.walk(file_path + symbol + document + symbol + X3Resolution):
#                 self.inputs = self.inputs + files
#         self.len = len(self.ouput)
#
# class X4Data(Dataset):
#     def __init__(self):
#         file_path = "benchmark"
#         documents = ["B100", "Set5", "Set14", "Urban100"]
#         symbol = "\\"
#         TruthValue = "HR"
#         X4Resolution = "LR_bicubic\\X4"
#         self.ouput = []
#         self.inputs = []
#         for document in documents:
#             for root, dirs, files in os.walk(file_path + symbol + document + symbol + TruthValue):
#                 self.ouput = self.ouput + files
#             for root, dirs, files in os.walk(file_path + symbol + document + symbol + X4Resolution):
#                 self.inputs = self.inputs + files
#         self.len = len(self.ouput)
#
#
#
# class AllData(Dataset):
#     def __init__(self):
#         file_path = "benchmark"
#         documents = ["B100", "Set5", "Set14", "Urban100"]
#         symbol = "\\"
#         TruthValue = "HR"
#         X2Resolution = "LR_bicubic\\X2"
#         X3Resolution = "LR_bicubic\\X3"
#         X4Resolution = "LR_bicubic\\X4"
#         self.ouput = []
#         self.inputs = []
#         for document in documents:
#             for root, dirs, files in os.walk(file_path + symbol + document + symbol + TruthValue):
#                 self.ouput = self.ouput + files
#             for root, dirs, files in os.walk(file_path + symbol + document + symbol + X2Resolution):
#                 self.inputs = self.inputs + files
#         self.len = len(self.ouput)
