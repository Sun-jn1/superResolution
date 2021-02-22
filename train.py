from Data import train_loader,test_loader
from model import Net,ModelHelper
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from option import args
save=False
# save=True
device="cuda"
epo_num = 100

Model = Net.DRN(args)
Model = Model.to(device)
lrModel = ModelHelper.DownBlock(args,4)
# fixModel = ModelHelper.DownBlockplus(args,4)  #change
lrModel = lrModel.to(device)
# fixModel = fixModel.to(device) #change
criterion = nn.MSELoss().to(device)
# criterion = nn.L1Loss().to(device) #change
learn_rate = 0.01
optimizer = optim.SGD(Model.parameters(),learn_rate)
lroptimizer = optim.SGD(lrModel.parameters(),learn_rate)
# fixoptimizer = optim.SGD(fixModel.parameters(),learn_rate) #change
schedule1 = torch.optim.lr_scheduler.StepLR(lroptimizer,step_size=50,gamma=0.5)
# schedule2 = torch.optim.lr_scheduler.StepLR(fixoptimizer,step_size=50,gamma=0.5)
schedule3 = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
train_iter_loss=[]
test_iter_loss=[]
epo_train_loss=[]
epo_test_loss=[]

for epo in range(epo_num):
    train_loss = 0
    Model.train()
    if (epo + 1) % 1 == 0:
        print("第", epo, "轮迭代")
    for i,(input,target) in enumerate(train_loader):
        input=input.to(device)
        target=target.to(device)
        output = Model(input)
        midput = output[1]
        output=output[-1]
        optimizer.zero_grad()
        lroptimizer.zero_grad()
        # fixoptimizer.zero_grad() #change
        loss1 = criterion(output,target)
        lr = lrModel(output)
        # fixlr = fixModel(target) #change
        # fixloss1 = criterion(fixlr[-1],input) #change
        # fixloss2 = criterion(midput,fixlr[0]) #change
        loss2 = criterion(lr,input)
        loss = loss1 + loss2
        # loss = 2*loss1+loss2+2*fixloss1+2*fixloss2 #change
        loss.backward()
        optimizer.step()
        lroptimizer.step()
        # fixoptimizer.step()
        iter_loss = loss.item()
        train_iter_loss.append(iter_loss)
        train_loss+=iter_loss

    schedule1.step()
    # schedule2.step()
    schedule3.step()

    train_loss/=len(train_loader)
    print(train_loss)
    epo_train_loss.append(train_loss)

    test_loss = 0
    Model.eval()
    with torch.no_grad():
        for index,(input,target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)
            output = Model(input)
            output = output[-1]
            optimizer.zero_grad()
            loss = criterion(output,target)
            iter_loss = loss.item()
            test_iter_loss.append(iter_loss)
            test_loss+=iter_loss
            if (index==1 or index==2) and epo==epo_num-1 and save==True:
                a,b,c,d=input.shape
                for j in range(a):
                    plt.imshow(input[j].cpu().detach().numpy().transpose(1,2,0))
                    path = "result\\model\\"+str(index)+"_"+str(j)+"_input.png"
                    plt.savefig(path)

                    plt.imshow(output[j].cpu().detach().numpy().transpose(1, 2, 0))
                    path = "result\\model\\" + str(index) + "_" + str(j) + "_output.png"
                    plt.savefig(path)

                    plt.imshow(target[j].cpu().detach().numpy().transpose(1, 2, 0))
                    path = "result\\model\\" + str(index) + "_" + str(j) + "_target.png"
                    plt.savefig(path)

        test_loss/=len(test_loader)
        print(test_loss)
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

torch.save(Model, './model.pth')
plt.plot(range(epo_num),epo_train_loss)
plt.title("train loss")
plt.savefig("result\\yuan_train_loss.png")

plt.figure()
plt.plot(range(epo_num),epo_test_loss)
plt.title("test loss")
plt.savefig("result\\yuan_test_loss.png")
plt.show()

