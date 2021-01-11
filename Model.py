import torch.nn as nn

class Extract_features(nn.Module):
    def __init__(self):
        super(Extract_features,self).__init__()
        self.conv0_1 = nn.Conv2d(3, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn0_1 = nn.BatchNorm2d(512)
        self.conv0_2 = nn.Conv2d(512, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn0_2 = nn.BatchNorm2d(768)
        self.compression0 = nn.Conv2d(768, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn0_3 = nn.BatchNorm2d(64)
        self.pooling0_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv1_1 = nn.Conv2d(64,512,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.bn1_2=nn.BatchNorm2d(512)
        self.compression1 = nn.Conv2d(512,64,kernel_size=(1,1),stride=(1,1))
        self.bn1_3=nn.BatchNorm2d(64)
        self.pooling1_1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)

        self.con2_1 = nn.Conv2d(64,256,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.bn2_1 = nn.BatchNorm2d(256)
        self.con2_2 = nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.bn2_2 = nn.BatchNorm2d(256)
        self.compression2 = nn.Conv2d(256,128,kernel_size=(1,1),stride=(1,1))
        self.bn2_3 = nn.BatchNorm2d(128)
        self.pooling2_1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)

        self.con3_1 = nn.Conv2d(128, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.bn3_1 = nn.BatchNorm2d(512)
        self.con3_2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.bn3_2 = nn.BatchNorm2d(512)
        self.compression3 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pooling3_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.relu = nn.ReLU()

    def forward(self,x):
        out = []
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        x = self.compression0(x)
        x = self.bn0_3(x)
        x = self.relu(x)
        x = self.pooling0_1(x)

        out.append(x)

        x=self.conv1_1(x)
        x=self.bn1_1(x)
        x=self.relu(x)
        x=self.conv1_2(x)
        x=self.bn1_2(x)
        x=self.relu(x)
        x=self.compression1(x)
        x=self.bn1_3(x)
        x=self.relu(x)
        x=self.pooling1_1(x)
        out.append(x)

        x=self.con2_1(x)
        x=self.bn2_1(x)
        x=self.relu(x)
        x=self.con2_2(x)
        x=self.bn2_2(x)
        x=self.relu(x)
        x=self.compression2(x)
        x=self.bn2_3(x)
        x=self.relu(x)
        x=self.pooling2_1(x)

        out.append(x)

        x=self.con3_1(x)
        x=self.bn3_1(x)
        x=self.relu(x)
        x=self.con3_2(x)
        x=self.bn3_2(x)
        x=self.relu(x)
        x=self.compression3(x)
        x=self.bn3_3(x)
        x=self.relu(x)
        x=self.pooling3_1(x)

        out.append(x)

        return out

class Recovery_features(nn.Module):
    def __init__(self,get_features):
        super(Recovery_features,self).__init__()
        self.get_features =  get_features
        # self.deconv1 = nn.ConvTranspose2d(256,128,kernel_size=7,stride=2,output_padding=1,padding=2,dilation=1)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, output_padding=0, padding=2, dilation=1)
        self.bn1=nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,output_padding=0,padding=1,dilation=1)
        self.bn2=nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,output_padding=1,padding=1,dilation=1)
        self.bn3=nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1, dilation=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.out_layer = nn.Conv2d(32,3,kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.get_features(x)
        features1 = features[3]
        features2=features[2]
        features3 = features[1]
        features4 = features[0]

        x = self.bn1(self.relu(self.deconv1(features1)))

        x=x+features2
        x = self.bn2(self.relu(self.deconv2(x)))

        x=x+features3
        x = self.bn3(self.relu(self.deconv3(x)))
        x = x + features4
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.out_layer(x)
        x=self.sigmoid(x)

        return x



