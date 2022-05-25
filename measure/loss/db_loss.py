import torch.nn as nn
from typing import Dict, Tuple
from measure.loss.bce_loss import BceLoss
from measure.loss.dice_loss import DiceLoss
from collections import OrderedDict
from torch import Tensor


class DBLoss(nn.Module):
    def __init__(self, binaryLoss: Dict):
        super().__init__()
        self._binaryLoss = BceLoss(**binaryLoss)

    def __call__(self, pred: OrderedDict, batch: OrderedDict) -> Tuple:
        lossDict: OrderedDict = OrderedDict()
        binaryDist: Tensor = self._binaryLoss(pred['binaryMap'],
                                              batch['binaryMap'],
                                              batch['probMask'])
        lossDict.update(binaryLoss=binaryDist)
        loss = binaryDist
        return loss, lossDict
