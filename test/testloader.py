from config import se_eb3
from dataset.loader import DetDataset
import yaml

if __name__ == "__main__":
    config = se_eb3

    dataset = DetDataset(**config['train']['dataset'])

    for i in range(dataset.__len__()):
        dataset.__getitem__(i, isVisual=True)

    # train = DetLoader(**config['train']).build()
    # for data in train:
    #     print(data.keys())
    #     break
