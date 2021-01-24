import torch.nn as nn
import torch
class convblock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1,activation='prelu'):
        super(convblock,self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation=activation
        if self.activation!=None:
            self.act = nn.PReLU()
    def forward(self,x):
        x = self.conv(x)
        if self.activation!=None:
            x = self.act(x)
        return x

class deconvblock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, activation='prelu'):
        super(deconvblock,self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.activation=activation
        if self.activation!=None:
            self.act = nn.PReLU()
    def forward(self,x):
        x = self.deconv(x)
        if self.activation!=None:
            x = self.act(x)
        return x

class upblock(nn.Module):
    def __init__(self,num_filter, kernel_size=8, stride=4, padding=2,activation='prelu'):
        super(upblock,self).__init__()
        self.up_conv1 = deconvblock(num_filter, num_filter, kernel_size, stride, padding,activation)
        self.up_conv2 = convblock(num_filter, num_filter, kernel_size, stride, padding,activation)
        self.up_conv3 = deconvblock(num_filter, num_filter, kernel_size, stride, padding,activation)
    def forward(self,x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class downblock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, activation='prelu'):
        super(downblock, self).__init__()
        self.down_conv1 = convblock(num_filter, num_filter, kernel_size, stride, padding ,activation)
        self.down_conv2 = deconvblock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.down_conv3 = convblock(num_filter, num_filter, kernel_size, stride, padding, activation)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0
class D_upblock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, activation='prelu'):
        super(D_upblock, self).__init__()
        self.conv = convblock(num_filter * num_stages, num_filter, 1, 1, 0, activation)
        self.up_conv1 = deconvblock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.up_conv2 = convblock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.up_conv3 = deconvblock(num_filter, num_filter, kernel_size, stride, padding, activation)
    def forward(self,x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class D_downblock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, activation='prelu'):
        super(D_downblock, self).__init__()
        self.conv = convblock(num_filter * num_stages, num_filter, 1, 1, 0, activation)
        self.down_conv1 = convblock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.down_conv2 = deconvblock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.down_conv3 = convblock(num_filter, num_filter, kernel_size, stride, padding, activation)

    def forward(self,x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0
class Net(nn.Module):
    def __init__(self,num_channels, base_filter, feat, num_stages):
        super(Net, self).__init__()
        kernel = 6
        stride = 2
        padding = 2
        self.feat0 = convblock(num_channels, feat, 3, 1, 1, activation='prelu')
        self.feat1 = convblock(feat, base_filter, 1, 1, 0, activation='prelu')

        self.up1 = upblock(base_filter, kernel, stride, padding)
        self.down1 = downblock(base_filter, kernel, stride, padding)
        self.up2 = upblock(base_filter, kernel, stride, padding)
        self.down2 = D_downblock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_upblock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_downblock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_upblock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_downblock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_upblock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_downblock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_upblock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_downblock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_upblock(base_filter, kernel, stride, padding, 6)
        self.down7 = D_downblock(base_filter, kernel, stride, padding, 7)
        self.up8 = D_upblock(base_filter, kernel, stride, padding, 7)
        self.down8 = D_downblock(base_filter, kernel, stride, padding, 8)
        self.up9 = D_upblock(base_filter, kernel, stride, padding, 8)
        self.down9 = D_downblock(base_filter, kernel, stride, padding, 9)
        self.up10 = D_upblock(base_filter, kernel, stride, padding, 9)

        self.output_conv = convblock(num_stages * base_filter, num_channels, 3, 1, 1,activation=None)

    def forward(self,x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down7(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up8(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down8(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up9(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down9(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up10(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        x=self.output_conv(concat_h)
        # s = nn.Sigmoid()
        # x=s(x)
        return x







