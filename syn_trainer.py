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
from config import se_eb0, se_eb1, se_eb2, se_eb3, syn_se_eb3


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
                 save_interval: int,
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
        self._loss = 1000.0
        self._save_interval = save_interval
        self._totalLoss: DetAverager = DetAverager()
        self._probLoss: DetAverager = DetAverager()
        self._threshLoss: DetAverager = DetAverager()
        self._binaryLoss: DetAverager = DetAverager()

    def _updateLR(self):
        rate: float = (1. - self._step / 100000) ** self._factor
        self._curLR: float = rate * self._lr
        for groups in self._optim.param_groups:
            groups['lr'] = self._curLR

    def _load(self):
        stateDict: Tuple = self._checkpoint.load(self._device)
        if stateDict is not None:
            self._model.load_state_dict(stateDict[0])
            # self._optim.load_state_dict(stateDict[1])
            # self._step = stateDict[2]

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
            if self._step > 100000:
                break
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
            self._updateLR()
            if self._step % self._save_interval == 0:
                self._save({
                    'totalLoss': self._totalLoss.calc(),
                    'probLoss': self._probLoss.calc(),
                    'threshLoss': self._threshLoss.calc(),
                    'binaryLoss': self._binaryLoss.calc(),
                })
                self._totalLoss.reset()
                self._probLoss.reset()
                self._threshLoss.reset()
                self._binaryLoss.reset()

    def _save(self, trainRS: Dict):
        self._checkpoint.saveCheckpoint(self._step, self._model, self._optim)
        self._logger.reportTime("Step {}:".format(self._step))
        self._logger.reportMetric(" - Training", trainRS)
        self._logger.reportMetric(" - Min_loss", {"total_loss": self._loss})
        self._logger.writeFile({
            'training': trainRS
        })


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Training config")
    parser.add_argument("-d", '--data', default='', type=str, help="path of data")
    parser.add_argument("-i", '--imgType', default=0, type=int, help="type of image")
    parser.add_argument("-s", '--save_interval', default=150, type=int, help="number of step to save")
    parser.add_argument("-b", '--start_epoch', default=1, type=int, help="start epoch")
    parser.add_argument("-r", '--resume', default='', type=str, help="resume path")
    args = parser.parse_args()
    config = syn_se_eb3
    if args.data.strip():
        for item in ["train", "valid"]:
            config[item]['dataset']['imgDir'] = os.path.join(args.data.strip(), "image/")
            config[item]['dataset']['tarFile'] = os.path.join(args.data.strip(), "{}.json".format(item))
            config[item]['dataset']['imgType'] = args.imgType
    if args.resume.strip():
        config['checkpoint']['resume'] = args.resume.strip()
    config['startEpoch'] = args.start_epoch
    trainer = DetTrainer(**config, save_interval=args.save_interval)
    trainer.train()
