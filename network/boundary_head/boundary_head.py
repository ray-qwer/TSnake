import torch
import torch.nn as nn
import math

class boundaryHead(nn.Module):
    def __init__(self, in_c, out_c, dim, down_ratio,):
        super().__init__()
        # iters = int(math.log2(down_ratio))
        iters = 4
        modulelist = []
        channels = [dim] * (iters + 1)
        if iters > 0:
            channels[0] = in_c
            channels[-1] = out_c

        # no upsampling
        for i in range(iters):
            # modulelist.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            modulelist.append(nn.Conv2d(channels[i], channels[i+1], 3, padding=1))
            modulelist.append(nn.BatchNorm2d(channels[i+1]))
            modulelist.append(nn.ReLU(inplace=True))
        
        self.local_feat_conv = nn.Sequential(*modulelist)

        self.boundary_pred = nn.Sequential(
            nn.Conv2d(out_c, 1, 1),
        )
    def forward(self, x, is_training=False):
        x = self.local_feat_conv(x)
        if True:
            bd_pred = self.boundary_pred(x)
            return bd_pred
        else:
            return x
        