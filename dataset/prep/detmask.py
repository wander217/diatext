from typing import Dict
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
from collections import OrderedDict


class DetMask:
    def __init__(self, ignore_thresh: float, erode: int, dilate: int):
        self._ignore_thresh: float = ignore_thresh
        self._erode: int = erode
        self._dilate: int = dilate

    def __call__(self, data: Dict, isVisual: bool = False):
        """
        preprocessing input data with imgaug
        :param data: a dict contains: img, train, tar
        :return: data processing with imgaug: img, train, tar, anno, shape
        """
        output = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: Dict):
        cv.imshow('prob_map', data['probMap'])
        cv.imshow('prob_mask', data['probMask'])
        cv.imshow('thresh_map', data['threshMap'])
        cv.imshow('thresh_mask', data['threshMask'])

    def _build(self, data: Dict) -> Dict:
        img = data['img']
        prob = np.zeros(img.shape)
        prob_mask = np.ones(img.shape)
        thresh_mask = np.zeros(img.shape)
        ignore = np.zeros((len(data['target']),)).astype(np.bool)
        boxes = []
        for i, target in enumerate(data['target']):
            tmp = np.array(target['bbox'])
            boxes.append(tmp)
            polygon = Polygon(tmp)
            if not polygon.is_valid \
                    or not polygon.is_simple \
                    or target['text'] == "###" \
                    or polygon.area < self._ignore_thresh:
                ignore[i] = True
                cv.fillPoly(prob_mask, [tmp.astype(np.int32)], 0)
                continue
            cv.fillPoly(prob, [tmp.astype(np.int32)], 1)
            cv.fillPoly(thresh_mask, [tmp.astype(np.int32)], 1)
        prob_map = cv.dilate(prob, (self._erode, self._erode))
        thresh_map = prob_map - cv.erode(prob, (self._dilate, self._dilate))
        new_data = OrderedDict(img=img,
                               polygon=boxes,
                               ignore=ignore,
                               probMap=prob_map,
                               probMask=prob_mask,
                               threshMap=thresh_map,
                               threshMask=thresh_mask)
        return new_data
