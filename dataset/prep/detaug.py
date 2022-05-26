import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from typing import Dict, List, Tuple
import numpy as np
import math
import cv2 as cv


class DetAug:
    def __init__(self, onlyResize: bool, **kwargs):
        # loading module inside imgaug
        moduls: List = []
        for key, item in kwargs.items():
            module = getattr(iaa, key)
            if module is not None:
                moduls.append(module(**item))
        self._prep = None
        self._onlyResize: bool = onlyResize
        # creating preprocess sequent
        if len(moduls) != 0:
            self._prep = iaa.Sequential(moduls)

    def __call__(self, data: Dict, isVisual: bool = False):
        """
        preprocessing input data with imgaug
        :param data: a dict contains: img, train, tar
        :return: data processing with imgaug: img, train, tar, anno, shape
        """
        output = self._build(data)
        if isVisual:
            self._visual(data)
        return output

    def _visual(self, data: Dict, lineHeight: int = 2):
        img = data['img']
        tars = data['polygon']
        for tar in tars:
            cv.polylines(img,
                         [np.int32(tar).reshape((1, -1, 2))],
                         True,
                         (255, 255, 0),
                         lineHeight)
        cv.imshow('aug_visual', img)

    def _build(self, data: Dict) -> Dict:
        image: np.ndarray = data['img']
        shape: Tuple = image.shape

        if self._prep is not None:
            aug = self._prep.to_deterministic()
            data['img'] = self._resize(data['img']) if self._onlyResize else aug.augment_image(data['img'])
            data['probMap'] = self._resize(data['probMap']) if self._onlyResize else aug.augment_image(
                data['probMap'])
            data['probMask'] = self._resize(data['probMask']) if self._onlyResize else aug.augment_image(
                data['probMask'])
            data['threshMap'] = self._resize(data['threshMap']) if self._onlyResize else aug.augment_image(
                data['threshMap'])
            data['threshMask'] = self._resize(data['threshMask']) if self._onlyResize else aug.augment_image(
                data['threshMask'])
            self._makeAnnotation(aug, data, shape)
            # saving shape to recover
            data.update(orgShape=shape[:2])
        return data

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
              Resize image when valid/test
        """
        if len(image.shape) == 3:
            org_h, org_w, _ = image.shape
            new_image = np.zeros((960, 960, 3), dtype=np.uint8)
            new_image[:org_h, :org_w] = image
        else:
            org_h, org_w = image.shape
            new_image = np.zeros((960, 960), dtype=np.uint8)
            new_image[:org_h, :org_w] = image
        return new_image

    def _makeAnnotation(self, aug, data: Dict, shape: Tuple) -> Dict:
        """
           Changing bounding box coordinates
        """
        if aug is None:
            return data

        polygons: List = []
        if not self._onlyResize:
            for item in data['polygon']:
                keyPoints: List = [Keypoint(point[0], point[1]) for point in item]
                keyPoints = aug.augment_keypoints([
                    KeypointsOnImage(keyPoints, shape=shape)
                ])[0].keypoints
                newPolygon: List = [(keyPoint.x, keyPoint.y) for keyPoint in keyPoints]
                polygons.append(newPolygon)
            data['polygon'] = polygons
        return data
