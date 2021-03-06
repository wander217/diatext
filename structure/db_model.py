from collections import OrderedDict
from torch import nn, Tensor
from structure.backbone import DBEfficientNet
from structure.neck import DBNeck
from structure.head import DBHead
from typing import List, Dict
import torch
import time
import yaml
from config import se_eb3


class DBModel(nn.Module):
    def __init__(self, backbone: Dict, neck: Dict, head: Dict):
        super().__init__()
        self._backbone = DBEfficientNet(**backbone)
        self._neck = DBNeck(**neck)
        self._head = DBHead(**head)

    def forward(self, x: Tensor, training: bool = True) -> OrderedDict:
        brs: List = self._backbone(x)
        nrs: Tensor = self._neck(brs)
        hrs: OrderedDict = self._head(nrs, training=training)
        return hrs


# test
if __name__ == "__main__":
    data = se_eb3
    model = DBModel(**data['lossModel']['model'])
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:', total_params)
    print('train params:', train_params)
    a = torch.rand((1, 3, 640, 640), dtype=torch.float)
    start = time.time()
    b = model(a)
    print('run:', time.time() - start)
