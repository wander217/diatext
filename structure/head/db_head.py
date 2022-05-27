import torch
from torch import nn, Tensor
from collections import OrderedDict
from typing import List
import torch.nn.functional as F


class DBHead(nn.Module):
    def __init__(self, exp: int):
        super().__init__()

        exp_output: int = exp // 4
        self.prob: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, exp_output, kernel_size=2, stride=2),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, 1, kernel_size=2, stride=2),
            nn.Hardsigmoid(inplace=True)
        )

        self.thresh: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, exp_output, kernel_size=2, stride=2),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, 1, kernel_size=2, stride=2),
            nn.Hardsigmoid(inplace=True)
        )

    def resize(self, x: Tensor, shape: List):
        return F.interpolate(x, shape, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor, shape: List) -> OrderedDict:
        result: OrderedDict = OrderedDict()
        # calculate probability map
        probMap: Tensor = self.prob(x)
        threshMap:Tensor = self.thresh(x)
        binaryMap: Tensor = F.sigmoid(probMap - threshMap)
        result.update(probMap=self.resize(probMap, shape),
                      threshMap=self.resize(threshMap, shape),
                      binaryMap=self.resize(binaryMap, shape))
        return result
