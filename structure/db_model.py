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


def weight_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        init_range = 1 / math.sqrt(module.out_features)
        nn.init.uniform_(module.weight, -init_range, init_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DBModel(nn.Module):
    def __init__(self, asn: Dict, backbone: Dict, neck: Dict, head: Dict):
        super().__init__()
        self._asn = AdaptiveScaleNetwork(**asn)
        self._backbone = DBEfficientNet(**backbone)
        self._neck = DBNeck(**neck)
        self._head = DBHead(**head)
        self.apply(weight_init)

    def forward(self, x: Tensor) -> OrderedDict:
        asn: Tensor = self._asn(x)
        brs: List = self._backbone(asn)
        nrs: Tensor = self._neck(brs)
        hrs: OrderedDict = self._head(nrs)
        return hrs


# test
if __name__ == "__main__":
    file_config: str = r'D:\python_project\diatext\config\dbpp_se_eb3.yaml'
    with open(file_config) as stream:
        data = yaml.safe_load(stream)
    model = DBModel(**data['lossModel']['model'])
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:', total_params)
    print('train params:', train_params)
    image = cv2.imread(r"D:\python_project\dbpp\google_text\train\image\0a4cbebc7a6f8fba.jpg")
    h, w, c = image.shape
    new_image = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32, 3), dtype=np.uint8)
    new_image[:h, :w] = image
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    start = time.time()
    b = model(x.float())
    print('run:', time.time() - start)
    b = b['probMap'].squeeze().cpu().detach().numpy()
    cv2.imshow("abc", np.uint8(255 * b))
    cv2.waitKey(0)
