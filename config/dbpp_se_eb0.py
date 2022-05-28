se_eb0 = {
    "totalEpoch": 1000,
    "startEpoch": 1,
    "lr": 0.003,
    "factor": 0.9,
    "lossModel": {
        "model": {
            "backbone": {
                "netID": "b0",
                "depthProb": 0.2,
                "useSE": True
            },
            "neck": {
                "dataPoint": (
                    24,
                    40,
                    112,
                    1280
                ),
                "exp": 256
            },
            "head": {
                "k": 50,
                "exp": 256,
                "adaptive": True
            }
        },
        "loss": {
            "threshScale": 10,
            "threshLoss": {
                "eps": 1e-06
            },
            "probScale": 5,
            "probLoss": {
                "ratio": 3.0,
                "eps": 1e-06
            },
            "binaryScale": 1,
            "binaryLoss": {
                "eps": 1e-06
            }
        }
    },
    "score": {
        "totalBox": 1000,
        "edgeThresh": 5,
        "probThresh": 0.3,
        "scoreThresh": 0.7,
        "label": "binaryMap"
    },
    "accurancy": {
        "ignoreThresh": 0.5,
        "scoreThresh": 0.7,
        "accThresh": 0.5
    },
    "train": {
        "batchSize": 8,
        "numWorkers": 4,
        "dropLast": True,
        "shuffle": True,
        "pinMemory": False,
        "dataset": {
            "imgType": 1,
            "imgDir": "D:\\python_project\\diatext\\vintext\\train/image",
            "tarFile": "D:\\python_project\\diatext\\vintext\\train/target.json",
            "prep": {
                "DetAug": {
                    "onlyResize": False,
                    "Fliplr": {
                        "p": 0.5
                    },
                    "Affine": {
                        "rotate": (-10, 10),
                        "fit_output": True
                    },
                    "Resize": {
                        "size": (0.5, 3.0)
                    }
                },
                "DetCrop": {
                    "crop": True,
                    "minCropSize": 0.1,
                    "maxTries": 10,
                    "generalSize": (640, 640)
                },
                "DetForm": {
                    "shrinkRatio": 0.4
                },
                "ProbMaker": {
                    "shrinkRatio": 0.4,
                    "minTextSize": 8
                },
                "ThreshMaker": {
                    "expandRatio": 0.4,
                    "minThresh": 0.3,
                    "maxThresh": 0.7
                },
                "DetNorm": {
                    "mean": (122.67891434, 116.66876762, 104.00698793)
                },
                "DetFilter": {
                    "key": [
                        "polygon",
                        "shape",
                        "ignore",
                        "train"
                    ]
                }
            }
        }
    },
    "valid": {
        "batchSize": 1,
        "numWorkers": 4,
        "dropLast": False,
        "shuffle": False,
        "pinMemory": False,
        "dataset": {
            "imgType": 1,
            "imgDir": "D:\\python_project\\diatext\\vintext\\valid/image",
            "tarFile": "D:\\python_project\\diatext\\vintext\\valid/target.json",
            "prep": {
                "DetAug": {
                    "onlyResize": True,
                    "Resize": {
                        "size": {
                            "height": 960,
                            "width": 960
                        }
                    }
                },
                "DetCrop": {
                    "crop": False,
                    "minCropSize": 0.1,
                    "maxTries": 10,
                    "generalSize": (960, 960)
                },
                "DetForm": {
                    "shrinkRatio": 0.4
                },
                "ProbMaker": {
                    "shrinkRatio": 0.4,
                    "minTextSize": 8
                },
                "ThreshMaker": {
                    "expandRatio": 0.4,
                    "minThresh": 0.3,
                    "maxThresh": 0.7
                },
                "DetNorm": {
                    "mean": (122.67891434, 116.66876762, 104.00698793)
                },
                "DetFilter": {
                    "key": [
                        "train"
                    ]
                }
            }
        }
    },
    "optimizer": {
        "name": "Adam",
        "args": {
            "betas": (0.9, 0.999),
            "eps": 1e-08
        }
    },
    "checkpoint": {
        "workspace": "checkpoint",
        "resume": ""
    },
    "logger": {
        "workspace": "logger",
        "level": "INFO"
    }
}
