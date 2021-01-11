#test Image
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# img = Image.open("testImg.png")
# print(img.size)
# plt.imshow(img)
# img.thumbnail((32,32))
# print(img.size)
# # img = np.array(img)
# plt.figure()
# plt.imshow(img)
# plt.figure()
# img=img.resize((600, 396),Image.ANTIALIAS)
# print(img.size)
# plt.imshow(img)
# plt.show()
#test Data
from Data import train_loader,test_loader,data_mean,data_std
import matplotlib.pyplot as plt
import numpy as np
import torch


for i,(input,target) in enumerate(train_loader):



    # print(type(input))
    # print(input.shape)
    # print(type(target))
    # print(target.shape)

    # target=np.array(target,dtype="int")
    mean = (0.42477179, 0.43378679, 0.3698565)
    std = (0.19699529, 0.1823555, 0.1743905)
    t_mean = np.array(torch.FloatTensor(mean).view(3, 1, 1).expand(3, 360, 360))
    t_std = np.array(torch.FloatTensor(std).view(3, 1, 1).expand(3, 360, 360))
    input = np.array(input)
    img = input[0]*t_std+t_mean
    img = img.transpose(1, 2, 0)

    plt.imshow(img)


    target = np.array(target)
    img=target[0].transpose(1,2,0)
    plt.figure()
    plt.imshow(img)
    plt.show()