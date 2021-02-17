# from model import Net
from option import args
from model import ModelHelper
import torch
a = torch.rand((4,40,64,64))
modelA = ModelHelper.PrimaryCaps(num_capsules=10, in_channels=40, out_channels=32, kernel_size=9, num_routes=10*32*28*28)
modelC = ModelHelper.DigitCaps(num_capsules=3, num_routes=10*32*28*28, in_channels=1, out_channels=16)
modelD = ModelHelper.Decoder(input_width=64, input_height=64, input_channel=3)
# a = torch.rand((4,3,16,16))
# x,y,z,c = a.shape

# modelA = ModelHelper.PrimaryCaps(num_capsules=32, in_channels=80, out_channels=32, kernel_size=9, num_routes=32*4*4)
# modelC = ModelHelper.DigitCaps(num_capsules=80, num_routes=32 * 4 * 4, in_channels=32, out_channels=16)
# modelD = ModelHelper.Decoder(input_width=16, input_height=16, input_channel=80)
b=modelA(a)
print("bbbbbbbbbbbbbb",b.shape)
d=modelC(b)
print("dddddddd",d.shape)
e=modelD(d)
# modelD=Net.DRN(args)
# e = modelD(a)
print("eeeeeeee",e.shape)
# for i in e:
#     print("结果",i.shape)

# model = ModelHelper.DownBlock(args,4)
# output = model(a)
# print(output.shape)
# if c == 32:
#     modelA = ModelHelper.PrimaryCaps(num_capsules=10, in_channels=80, out_channels=32, kernel_size=9,
#                                      num_routes=32 * 4 * 4 * 3 * 3)
#     modelC = ModelHelper.DigitCaps(num_capsules=3, num_routes=32 * 4 * 4 * 3 * 3, in_channels=10,
#                                    out_channels=16)
#     modelD = ModelHelper.Decoder(input_width=32, input_height=32, input_channel=3)
#     print(idx + 1, "tail前", x.shape)
#     sr = modelD(modelC(modelA(x)))
#     print(idx + 1, "tail后", sr.shape)
#     results.append(sr)
# print("tail前",x.shape)
#         # sr = self.tail[0](x)
#         modelA = ModelHelper.PrimaryCaps(num_capsules=3, in_channels=80, out_channels=32, kernel_size=9, num_routes=32*4*4*3)
#         modelC = ModelHelper.DigitCaps(num_capsules=3, num_routes=32*4*4*3, in_channels=1, out_channels=16)
#         modelD = ModelHelper.Decoder(input_width=16, input_height=16, input_channel=3)
#         sr = modelD(modelC(modelA(x)))
#         # print(x.shape)
#         print("tail后",sr.shape)
