import torch
from torch import nn, Tensor
from typing import List
from structure.neck.hour_glass import HourGlass


class DBNeck(nn.Module):
    def __init__(self, data_point: tuple, exp: int, layer_num: int, bias: bool = False):
        super().__init__()
        self._ins: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(data_point[i], exp, kernel_size=1, bias=bias)
            for i in range(len(data_point))
        ])
        self._hour_glass: nn.Module = nn.Sequential(*[
            HourGlass(exp, exp) for _ in range(layer_num)
        ])
        exp_output: int = exp // 4
        self.out5: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='bilinear')
        )
        self.out4: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        self.out3: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.out2: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, feature: List) -> Tensor:
        """
        :param feature: 4 feature with different size: 1/32, 1/16, 1/8, 1/4
        :return: primitive probability map
        """
        # input processing
        for i in range(len(self._ins)):
            feature[i] = self._ins[i](feature[i])
        # up sampling processing
        feature = self._hour_glass(feature)
        f2, f3, f4, f5 = feature
        f2 = self.out2(f2)
        f3 = self.out3(f3)
        f4 = self.out4(f4)
        f5 = self.out5(f5)
        out = torch.cat([f2, f3, f4, f5], dim=1)
        return out
