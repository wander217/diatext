import math
import numpy as np
import cv2 as cv
from typing import List, Dict


class DetCrop:
    def __init__(self, generalSize: List, maxTries: int, minCropSize: float):
        self._maxTries: int = maxTries
        self._minCropSize: float = minCropSize
        self._generalSize: List = generalSize

    def __call__(self, data: Dict, isVisual: bool = False) -> Dict:
        output: Dict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: Dict):
        cv.imshow('image', data['img'])
        cv.imshow('prob_map', data['probMap'])
        cv.imshow('prob_mask', data['probMask'])
        cv.imshow('thresh_map', data['threshMap'])
        cv.imshow('thresh_mask', data['threshMask'])

    def _build(self, data: Dict) -> Dict:
        img: np.ndarray = data['img']
        polygons: List = [polygon for polygon, ignore in zip(data['polygon'], data['ignore']) if not ignore]
        cropX, cropY, cropW, cropH = self._cropArea(img, polygons)
        scaleW: float = self._generalSize[0] / cropW
        scaleH: float = self._generalSize[1] / cropH
        scale: float = min(scaleH, scaleW)
        h = int(scale * cropH)
        w = int(scale * cropW)

        pad_image: np.ndarray = np.zeros((self._generalSize[1], self._generalSize[0], img.shape[2]), img.dtype)
        pad_image[:h, :w] = cv.resize(img[cropY:cropY + cropH, cropX:cropX + cropW], (w, h))
        pad_prob_map: np.ndarray = np.zeros((self._generalSize[1], self._generalSize[0], img.shape[2]), img.dtype)
        pad_prob_map[:h, :w] = cv.resize(data['probMap'][cropY:cropY + cropH, cropX:cropX + cropW], (w, h))
        pad_prob_mask: np.ndarray = np.zeros((self._generalSize[1], self._generalSize[0], img.shape[2]), img.dtype)
        pad_prob_mask[:h, :w] = cv.resize(data['probMask'][cropY:cropY + cropH, cropX:cropX + cropW], (w, h))
        pad_thresh_map: np.ndarray = np.zeros((self._generalSize[1], self._generalSize[0], img.shape[2]), img.dtype)
        pad_thresh_map[:h, :w] = cv.resize(data['threshMap'][cropY:cropY + cropH, cropX:cropX + cropW], (w, h))
        pad_thresh_mask: np.ndarray = np.zeros((self._generalSize[1], self._generalSize[0], img.shape[2]), img.dtype)
        pad_thresh_mask[:h, :w] = cv.resize(data['threshMask'][cropY:cropY + cropH, cropX:cropX + cropW], (w, h))

        new_polygons: List = []
        ignores: List = []
        for i, target in enumerate(data['polygon']):
            polygon = np.array(target)
            if not self._isOutside(polygon, [cropX, cropY, cropX + cropW, cropY + cropH]) \
                    and not data['ignore'][i]:
                new_polygon: List = ((polygon - (cropX, cropY)) * scale).tolist()
                new_polygons.append(new_polygon)
                ignores.append(False)
        data['polygon'] = new_polygons
        data['ignore'] = np.array(ignores)
        data['img'] = pad_image
        data['prob_map'] = pad_prob_map
        data['prob_mask'] = pad_prob_mask
        data['thresh_map'] = pad_thresh_map
        data['thresh_mask'] = pad_thresh_mask
        return data

    def _cropArea(self, img: np.ndarray, polygons: List) -> tuple:
        '''
            Hàm thực hiện cắt ảnh.
        '''
        h, w, _ = img.shape
        yAxis: np.ndarray = np.zeros(h, dtype=np.int32)
        xAxis: np.ndarray = np.zeros(w, dtype=np.int32)

        for polygon in polygons:
            tmp: np.ndarray = np.round(polygon, decimals=0).astype(np.int32)
            xAxis = self._maskDown(xAxis, tmp, 0)
            yAxis = self._maskDown(yAxis, tmp, 1)

        yNotMask: np.ndarray = np.where(yAxis == 0)[0]
        xNotMask: np.ndarray = np.where(xAxis == 0)[0]
        if len(xNotMask) == 0 or len(yNotMask) == 0:
            return 0, 0, w, h
        xSegment: List = self._splitRegion(xNotMask)
        ySegment: List = self._splitRegion(yNotMask)
        wMin: float = self._minCropSize * w
        hMin: float = self._minCropSize * h
        for _ in range(self._maxTries):
            xMin, xMax = self._choice(xSegment, xNotMask, w)
            yMin, yMax = self._choice(ySegment, yNotMask, h)
            newW = xMax - xMin + 1
            newH = yMax - yMin + 1
            if newW < wMin or newH < hMin:
                continue
            for polygon in polygons:
                if not self._isOutside(polygon, [xMin, yMin, xMax, yMax]):
                    return xMin, yMin, newW, newH
        return 0, 0, w, h

    def _choice(self, segment: List, axis: np.ndarray, limit: int):
        if len(segment) > 1:
            id_list: List = list(np.random.choice(len(segment), size=2))
            value_list: List = []
            for id in id_list:
                region: int = segment[id]
                x: int = int(np.random.choice(region, 1))
                value_list.append(x)
            x_min = np.clip(min(value_list), 0, limit - 1)
            x_max = np.clip(max(value_list), 0, limit - 1)
        else:
            x_list: np.ndarray = np.random.choice(axis, size=2)
            x_min = np.clip(np.min(x_list), 0, limit - 1)
            x_max = np.clip(np.max(x_list), 0, limit - 1)
        return x_min, x_max

    def _maskDown(self, axis: np.ndarray, polygon: np.ndarray, type: int) -> np.ndarray:
        '''
            masking axis by a byte
            type =1 : y axis, type=0 : x axis
        '''
        p_axis: np.ndarray = polygon[:, type]
        minValue: int = np.min(p_axis)
        maxValue: int = np.max(p_axis)
        axis[minValue:maxValue + 1] = 1
        return axis

    def _splitRegion(self, axis: np.ndarray) -> List:
        '''
            splitting by axis
        '''
        region: List = []
        startPoint: int = 0
        if axis.shape[0] == 0:
            return region
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region.append(axis[startPoint:i])
                startPoint = i
        if startPoint < axis.shape[0]:
            region.append(axis[startPoint:])
        return region

    def _isOutside(self, polygon: np.ndarray, lim: List) -> bool:
        '''

        :param polygon: polygon is surrounding text, size: 4x2
        :param lim: limit of crop size: [xMin, yMin, xMax, yMax]
        :return: true/false
        '''
        tmp: np.ndarray = np.array(polygon)
        x_min = tmp[:, 0].min()
        x_max = tmp[:, 0].max()
        y_min = tmp[:, 1].min()
        y_max = tmp[:, 1].max()
        if x_min >= lim[0] and x_max <= lim[2] and y_min >= lim[1] and y_max <= lim[3]:
            return False
        return True
