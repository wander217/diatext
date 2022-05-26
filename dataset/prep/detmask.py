from typing import Dict
import cv2 as cv
import numpy as np
import pyclipper
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
        pass

    def _build(self, data: Dict) -> Dict:
        img = data['img']
        prob = np.zeros(img.shape[:2])
        prob_map = np.zeros(img.shape[:2])
        prob_mask = np.zeros(img.shape[:2])
        thresh_mask = np.zeros(img.shape[:2])
        ignore = np.zeros((len(data['target']),)).astype(np.bool)
        boxes = []
        for i, target in enumerate(data['target']):
            tmp = np.array(target['bbox'])
            polygon = Polygon(tmp)
            dist = polygon.area * (1 - np.power(0.4, 2)) / polygon.length
            subject = [tuple(point) for point in tmp]
            shrinking = pyclipper.PyclipperOffset()
            shrinking.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinkPolygon = shrinking.Execute(-dist)
            if len(shrinkPolygon) == 0:
                ignore[i] = True
                cv.fillPoly(prob_mask, [tmp.astype(np.int32)], 1)
                continue
            boxes.append(tmp)
            polygon = Polygon(shrinkPolygon[0])
            if not polygon.is_valid \
                    or not polygon.is_simple \
                    or target['text'] == "###" \
                    or polygon.area < self._ignore_thresh:
                ignore[i] = True
                cv.fillPoly(prob_mask, [tmp.astype(np.int32)], 1)
                continue
            cv.fillPoly(prob, [np.array(shrinkPolygon[0]).astype(np.int32)], 1)
            cv.fillPoly(prob_map, [np.array(tmp).astype(np.int32)], 1)
            cv.fillPoly(thresh_mask, [tmp.astype(np.int32)], 1)
        thresh_map = cv.dilate(prob_map - prob, np.ones((self._erode, self._erode)))
        new_data = OrderedDict(img=img,
                               polygon=boxes,
                               ignore=ignore,
                               probMap=prob_map,
                               probMask=prob_mask,
                               threshMap=thresh_map,
                               threshMask=thresh_mask)
        return new_data
