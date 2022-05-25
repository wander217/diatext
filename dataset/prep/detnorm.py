import numpy as np
from typing import List, OrderedDict


class DetNorm:
    def __init__(self, mean: List):
        self.mean: np.ndarray = np.array(mean)

    def __call__(self, data: OrderedDict, isVisual: bool = False):
        output: OrderedDict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: OrderedDict):
        print(data.keys())
        print(data['img'].shape)
        # threshMap = np.uint8(data['threshMap'] * 255)
        # threshMask = np.uint8(data['threshMask'] * 255)
        # cv2.imshow("new thresh mask", threshMask)
        # cv2.imshow("new thresh map", threshMap)
        # probMap = np.uint8(data['probMap'][0] * 255)
        # probMask = np.uint8(data['probMask'] * 255)
        # cv2.imshow("new prob Map", probMap)
        # cv2.imshow("new prob Mask", probMask)

    def _build(self, data: OrderedDict) -> OrderedDict:
        assert 'img' in data
        image: np.ndarray = data['img']
        image = (image.astype(np.float64) - self.mean) / 255.
        data['img'] = np.transpose(image, (2, 0, 1))
        return data
