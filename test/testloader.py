from dataset import DetLoader
from dataset.loader import DetDataset
from collections import OrderedDict
import yaml

if __name__ == "__main__":
    configPath = r'D:\python_project\dbpp\config\dbpp_se_eb0.yaml'

    with open(configPath, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset = DetDataset(**config['train']['dataset'])

    for i in range(dataset.__len__()):
        dataset.__getitem__(i, isVisual=True)

    # train = DetLoader(**config['train']).build()
    # for data in train:
    #     print(data.keys())
    #     break
