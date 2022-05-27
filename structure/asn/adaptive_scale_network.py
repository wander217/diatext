import math
import time
import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import List
import torch.nn.functional as F


def resize(x: Tensor, org_size: List):
    return F.interpolate(x, org_size, mode='bilinear', align_corners=False)


class AdaptiveScaleNetwork(nn.Module):
    def __init__(self, shape: List, hidden_channel: int):
        super().__init__()
        self._shape: List = shape
        self._weight_init: nn.Module = nn.Sequential(
            nn.Conv2d(3, hidden_channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_channel)
        )
        self._residual: nn.Module = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_channel),
            nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1),
            nn.BatchNorm2d(hidden_channel))
        self._conv: nn.Module = nn.Sequential(
            nn.Conv2d(hidden_channel, 1, 3, 1, 1)
        )

    def forward(self, x: Tensor):
        y: Tensor = resize(self._weight_init(x), self._shape)
        y = self._residual(y) + y
        y = self._conv(y)
        return y


if __name__ == "__main__":
    conv = AdaptiveScaleNetwork([640, 640], 16)
    img = cv2.imread(r'D:\python_project\dbpp\google_text\train\image\0a4cbebc7a6f8fba.jpg')
    h, w, c = img.shape
    new_image = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32, 3), dtype=np.uint8)
    new_image[:h, :w] = img
    new_h, new_w, _ = new_image.shape
    input = torch.from_numpy(new_image).permute(2, 0, 1).unsqueeze(0)
    start = time.time()
    new_image = conv(input.float()).squeeze().cpu().detach().numpy()
    # new_image = cv2.resize(np.uint8(new_image), (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(np.uint8(img), (640, 640), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('aaa', np.uint8(new_image))
    cv2.imshow('bbb', np.uint8(img))
    cv2.waitKey(0)
    total_params = sum(p.numel() for p in conv.parameters())
    print(total_params)
    print(time.time() - start)
