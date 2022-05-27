from collections import OrderedDict
import cv2
import numpy as np
from torch import nn, Tensor
from structure.backbone import DBEfficientNet
from structure.neck import DBNeck
from structure.head import DBHead
from structure.asn import AdaptiveScaleNetwork
from typing import List, Dict
import torch
import time
import yaml
import math


class DBModel(nn.Module):
    def __init__(self, backbone: Dict, neck: Dict, head: Dict):
        super().__init__()
        self._backbone = DBEfficientNet(**backbone)
        self._neck = DBNeck(**neck)
        self._head = DBHead(**head)

    def forward(self, x: Tensor) -> OrderedDict:
        brs: List = self._backbone(x)
        nrs: Tensor = self._neck(brs)
        hrs: OrderedDict = self._head(nrs)
        return hrs


# test
if __name__ == "__main__":
    file_config: str = r'D:\db_pp\config\dbpp_se_eb3.yaml'
    with open(file_config) as stream:
        data = yaml.safe_load(stream)
    model = DBModel(**data['lossModel']['model'])
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:', total_params)
    print('train params:', train_params)
    a = torch.rand((1, 3, 640, 640), dtype=torch.float)
    start = time.time()
    b = model(a)
    print('run:', time.time() - start)
