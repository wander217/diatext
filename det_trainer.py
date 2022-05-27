import os.path
import yaml
import torch
from dataset import DetLoader
from tool import DetLogger, DetAverager, DetCheckpoint
from typing import Dict, Tuple
import torch.optim as optim
import argparse
import warnings
from loss_model import LossModel
from measure import DetAcc, DetScore


class DetTrainer:
    def __init__(self,
                 lossModel: Dict,
                 train: Dict,
                 valid: Dict,
                 optimizer: Dict,
                 accurancy: Dict,
                 score: Dict,
                 checkpoint: Dict,
                 logger: Dict,
                 totalEpoch: int,
                 startEpoch: int,
                 lr: float,
                 factor: float,
                 **kwargs):
        self._device = torch.device('cpu')
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        self._model: LossModel = LossModel(**lossModel, device=self._device)
        self._train = DetLoader(**train).build()
        self._valid = DetLoader(**valid).build()
        self._checkpoint: DetCheckpoint = DetCheckpoint(**checkpoint)
        self._acc: DetAcc = DetAcc(**accurancy)
        self._score: DetScore = DetScore(**score)
        self._logger: DetLogger = DetLogger(**logger)
        optimCls = getattr(optim, optimizer['name'])
        self._lr: float = lr
        self._factor: float = factor
        self._optim: optim.Optimizer = optimCls(**optimizer['args'],
                                                lr=self._lr,
                                                params=self._model.parameters())
        self._totalEpoch: int = totalEpoch + 1
        self._startEpoch: int = startEpoch
        self._curLR: float = lr
        self._step = 0
        self._f1score = 0.0
        self._totalLoss: DetAverager = DetAverager()
        self._probLoss: DetAverager = DetAverager()
        self._threshLoss: DetAverager = DetAverager()
        self._binaryLoss: DetAverager = DetAverager()

    def _updateLR(self, epoch: int):
        rate: float = (1. - epoch / self._totalEpoch) ** self._factor
        self._curLR: float = rate * self._lr
        for groups in self._optim.param_groups:
            groups['lr'] = self._curLR

    def _load(self):
        stateDict: Tuple = self._checkpoint.load(self._device)
        if stateDict is not None:
            self._model.load_state_dict(stateDict[0])
            # self._optim.load_state_dict(stateDict[1])
            # self._startEpoch = stateDict[2] + 1

    def train(self):
        self._load()
        self._logger.reportDelimitter()
        self._logger.reportTime("Start")
        self._logger.reportDelimitter()
        self._logger.reportNewLine()
        for i in range(self._startEpoch, self._totalEpoch):
            self._logger.reportDelimitter()
            self._logger.reportTime("Epoch {}".format(i))
            self._trainStep()
        self._logger.reportDelimitter()
        self._logger.reportTime("Finish")
        self._logger.reportDelimitter()

    def _trainStep(self):
        self._model.train()
        for i, batch in enumerate(self._train):
            self._optim.zero_grad()
            batchSize: int = batch['img'].size(0)
            pred, loss, metric = self._model(batch)
            loss = loss.mean()
            loss.backward()
            self._optim.step()
            self._totalLoss.update(loss.item() * batchSize, batchSize)
            self._probLoss.update(metric['probLoss'].item() * batchSize, batchSize)
            self._threshLoss.update(metric['threshLoss'].item() * batchSize, batchSize)
            self._binaryLoss.update(metric['binaryLoss'].item() * batchSize, batchSize)
            self._step += 1
            if self._step % 5 == 0:
                validRS = self._validStep()
                self._model.train()
                self._save({
                    'totalLoss': self._totalLoss.calc(),
                    'probLoss': self._probLoss.calc(),
                    'threshLoss': self._threshLoss.calc(),
                    'binaryLoss': self._binaryLoss.calc(),
                }, validRS)
                self._totalLoss.reset()
                self._probLoss.reset()
                self._threshLoss.reset()
                self._binaryLoss.reset()

    def _validStep(self) -> Dict:
        self._model.eval()
        totalLoss: DetAverager = DetAverager()
        threshLoss: DetAverager = DetAverager()
        probLoss: DetAverager = DetAverager()
        binaryLoss: DetAverager = DetAverager()
        precision: DetAverager = DetAverager()
        recall: DetAverager = DetAverager()
        f1score: DetAverager = DetAverager()
        with torch.no_grad():
            for batch in self._valid:
                batchSize: int = batch['img'].size(0)
                pred, loss, metric = self._model(batch)
                totalLoss.update(loss.mean().item() * batchSize, batchSize)
                probLoss.update(metric['probLoss'].item() * batchSize, batchSize)
                threshLoss.update(metric['threshLoss'].item() * batchSize, batchSize)
                binaryLoss.update(metric['binaryLoss'].item() * batchSize, batchSize)
                self._acc(*self._score(pred, batch), batch)
                measure = self._acc.gather()
                # print(measure)
                precision.update(measure['precision'] * batchSize, batchSize)
                recall.update(measure['recall'] * batchSize, batchSize)
                f1score.update(measure['f1score'] * batchSize, batchSize)
        return {
            'totalLoss': totalLoss.calc(),
            'probLoss': probLoss.calc(),
            'threshLoss': threshLoss.calc(),
            'binaryLoss': binaryLoss.calc(),
            'precision': precision.calc(),
            'recall': recall.calc(),
            'f1score': f1score.calc(),
        }

    def _save(self, trainRS: Dict, validRS: Dict):
        if validRS['f1score'] > self._f1score:
            self._f1score = validRS['f1score']
            self._checkpoint.saveCheckpoint(self._step, self._model, self._optim)
        self._logger.reportTime("Step {}:".format(self._step))
        self._logger.reportMetric(" - Training", trainRS)
        self._logger.reportMetric(" - Validation", validRS)
        self._logger.reportMetric(" - Best", {"f1score": self._f1score})
        self._logger.writeFile({
            'training': trainRS,
            'validation': validRS,
            "best": self._f1score
        })
        self._checkpoint.saveModel(self._model, self._step)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Training config")
    parser.add_argument("-p", '--path', type=str, help="path of config file")
    parser.add_argument("-d", '--data', default='', type=str, help="path of data")
    parser.add_argument("-i", '--imgType', default=0, type=int, help="type of image")
    parser.add_argument("-r", '--resume', default='', type=str, help="resume path")
    args = parser.parse_args()
    with open(args.path) as f:
        config: Dict = yaml.safe_load(f)
    if args.data.strip():
        for item in ["train", "valid"]:
            config[item]['dataset']['imgDir'] = os.path.join(args.data.strip(), item, "image/")
            config[item]['dataset']['tarFile'] = os.path.join(args.data.strip(), item, "target.json")
            config[item]['dataset']['imgType'] = args.imgType
    if args.resume.strip():
        config['checkpoint']['resume'] = args.resume.strip()
    trainer = DetTrainer(**config)
    trainer.train()
