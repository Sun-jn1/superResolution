import torch.nn.functional as F
import torch
from Data import pdata
from matplotlib import pyplot as plt
import numpy as np
model = torch.load('./model.pth')
def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
for index, (input, target) in enumerate(pdata):
    input = input.to("cuda")
    target = target.to("cuda")
    output = model(input)
    output = output[-1]
    a,b,c,d=input.shape
    for i in range(a):

        input1 = input[i].cpu()

        input1 = input1.detach().numpy()
        plt.imshow(input1.transpose(1, 2, 0))

        m = input1.transpose(1, 2, 0)

        plt.imsave("result\\dataShow\\"+str(index)+"_"+str(i)+"input.jpg",m)
        plt.figure()
        output1 = output[i].cpu()
        output1 = output1.detach().numpy()
        plt.imshow(output1.transpose(1, 2, 0))
        m=output1.transpose(1, 2, 0)
        # m=np.round(np.abs(m),1)
        m=normalize(m)
        plt.imsave("result\\dataShow\\" + str(index) + "_" + str(i) + "output.jpg", m)
        plt.figure()
        target1 = target[i].cpu()
        target1 = target1.detach().numpy()
        plt.imshow(target1.transpose(1, 2, 0))
        m=target1.transpose(1, 2, 0)
        plt.imsave("result\\dataShow\\" + str(index) + "_" + str(i) + "target.jpg", m)
    plt.show()
    break