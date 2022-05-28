import json
import os
import time
from loss_model import LossModel
import torch
import yaml

from measure import DetAcc
from measure.metric import DetScore
from typing import Dict, List, Tuple, OrderedDict
import numpy as np
import cv2 as cv
from config import se_eb3


class DBPredictor:
    def __init__(self, config: Dict, pretrained):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self._model = LossModel(**config['lossModel'], device=self.device)
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        state_dict = torch.load(pretrained, map_location=self.device)
        self._model.load_state_dict(state_dict['model'])
        # multi scale problem => training
        self._score: DetScore = DetScore(**config['score'])
        self._limit: int = 960

    def _resize(self, image: np.ndarray) -> Tuple:
        org_h, org_w, _ = image.shape
        scale = min([self._limit / org_h, self._limit / org_w])
        new_h = int(scale * org_h)
        new_w = int(scale * org_w)
        new_image = np.zeros((self._limit, self._limit, 3), dtype=np.uint8)
        image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        new_image[:new_h, :new_w, :] = image
        return new_image, new_h, new_w

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        mean = [122.67891434, 116.66876762, 104.00698793]
        image = image.astype(np.float64)
        image = (image - mean) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image.unsqueeze(0)

    def __call__(self, image: np.ndarray) -> Tuple:
        self._model.eval()
        bboxes: List = []
        scores: List = []
        with torch.no_grad():
            h, w, _ = image.shape
            reImage, newH, newW = self._resize(image)
            inputImage = self._normalize(reImage)
            pred: OrderedDict = self._model(dict(img=inputImage), training=False)
            bs, ss = self._score(pred, dict(img=inputImage))

            for i in range(len(bs[0])):
                if ss[0][i] > 0:
                    # bboxes.append(bs[0][i])
                    bboxes.append(bs[0][i] * np.array([w / newW, h / newH]))
                    scores.append(ss[0][i])
            return np.array(bboxes), np.array(scores)


if __name__ == "__main__":
    pretrainedPath: str = r'D:\workspace\project\diatext\checkpoint_27450.pth'
    predictor = DBPredictor(se_eb3, pretrainedPath)
    root: str = r'D:\icdar15\valid'
    count = 0
    precision, recall, f1score = 0, 0, 0
    for subRoot, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".jpg"):
                img = cv.imread(os.path.join(subRoot, file))
                boxes, scores = predictor(img)
                # for item in boxes:
                #     cv.polylines(img, [item.astype(np.int32)], True, (0, 255, 0))
                # cv.imshow("abc", img)
                # cv.waitKey(0)
                with open(r"D:\icdar15\valid\target.json", encoding='utf-8') as f:
                    data = json.loads(f.readline())
                gt = {
                    "polygon": [[]],
                    "ignore": [[]]
                }
                for item in data:
                    if item['file_name'] == file:
                        for bbox in item['target']:
                            gt['polygon'][0].append(bbox['bbox'])
                            gt['ignore'][0].append(False)
                det_acc = DetAcc(0.5, 0.5, 0.3)
                det_acc(boxes[np.newaxis, :], scores[np.newaxis, :], gt)
                result = det_acc.gather()
                print(result)
                result['file_name'] = "test{}.jpg".format(count)
                recall += result['recall']
                precision += result['precision']
                f1score += result['f1score']
                count += 1
    print("recall: {}, precision: {}, f1score: {}".format(recall / count,
                                                          precision / count,
                                                          f1score / count))
