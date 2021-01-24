from Data import train_loader,test_loader
from DBPNmodel import Net
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

device="cuda"
epo_num = 100

Model = Net(num_channels=3, base_filter=64, feat = 256, num_stages=10)
Model = Model.to(device)
criterion = nn.MSELoss().to(device)
learn_rate = 0.1
optimizer = optim.SGD(Model.parameters(),learn_rate)
train_iter_loss=[]
test_iter_loss=[]
epo_train_loss=[]
epo_test_loss=[]

for epo in range(epo_num):
    learn_rate = learn_rate*(0.1**((epo+1)%100))
    train_loss = 0
    Model.train()
    if (epo + 1) % 1 == 0:
        print("第", epo, "轮迭代")
    for i,(input,target) in enumerate(train_loader):
        # target = torch.sigmoid(target)
        input=input.to(device)
        target=target.to(device)
        output = Model(input)

        optimizer.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        iter_loss = loss.item()
        train_iter_loss.append(iter_loss)
        train_loss+=iter_loss
        optimizer.step()
    epo_train_loss.append(train_loss)

    test_loss = 0
    Model.eval()
    with torch.no_grad():
        for index,(input,target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)
            output = Model(input)
            optimizer.zero_grad()
            loss = criterion(output,target)
            iter_loss = loss.item()
            test_iter_loss.append(iter_loss)
            test_loss+=iter_loss
        epo_test_loss.append(test_loss)


    if epo==epo_num-1:
        input1 = input[0].cpu()
        input1 = input1.detach().numpy()
        plt.imshow(input1.transpose(1,2,0))
        plt.figure()
        output1 = output[0].cpu()

        output1 = output1.detach().numpy()

        plt.imshow(output1.transpose(1,2,0))
        plt.figure()
        target1 = target[0].cpu()
        target1 = target1.detach().numpy()

        plt.imshow(target1.transpose(1, 2, 0))
        plt.figure()


plt.plot(range(epo_num),epo_train_loss)
plt.title("train loss")
plt.figure()
plt.plot(range(epo_num),epo_test_loss)
plt.title("test loss")
plt.show()

