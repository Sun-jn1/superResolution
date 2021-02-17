import torch
import torch.nn as nn
import ModelHelper
class DRN(nn.Module):
    def __init__(self, opt, conv=ModelHelper.default_conv):
        super(DRN, self).__init__()
        ###一会需要设置为原模型进行测试
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)

        n_blocks = opt.n_blocks
        n_feats = opt.n_feats

        kernel_size = 3

        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = ModelHelper.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            ModelHelper.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            ModelHelper.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            ModelHelper.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            ModelHelper.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                ModelHelper.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), opt.n_colors, kernel_size)
            )
        # tail = [conv(n_feats, opt.n_colors, kernel_size)] #change
        # for p in range(self.phase, 0, -1):
        #     tail.append(
        #         conv(n_feats*pow(2, p), opt.n_colors, kernel_size)
        #     )                                             #end
        self.tail = nn.ModuleList(tail)

        # self.add_mean = ModelHelper.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        # upsample x to target sr size
        x = self.upsample(x)
        # print("upsample对x的处理",x.shape)
        # print(x.shape)
        # preprocess
        # x = self.sub_mean(x)
        x = self.head(x)
        # print("head对x的处理",x.shape)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)


        # up phases
        # print("tail前",x.shape)
        sr = self.tail[0](x)

        # print(x.shape)
        # print("tail后",sr.shape)

        # sr = self.add_mean(sr)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            # print("up_blocks 之前",x.shape)
            x = self.up_blocks[idx](x)
            # print("up_blocks 之后", x.shape)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # x = x+copies[self.phase - idx - 1]  #尝试更改
            # output sr imgs
            sr = self.tail[idx + 1](x)
            results.append(sr)
            # sr = self.add_mean(sr)



        return results