import sys
import os
import autorootcwd
import yaml
import argparse
import torch

from data.dataset import CustomedDataset_Seg, CustomedDataset_MultiClass_Seg, buildDataLoader
from data.dataset import RandomFlip, Normalization, ToTensor, ToOnehot, RandomCrop
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from src.models.created_model import MODEL_CLASSES
from src.utils.utils import load_config, Parser
#from scripts.train_seg import Trainer
from scripts.train_seg_ver2 import Trainer
from src.utils.utils import build_settings_segmentation


def nested_update(config, flat_dict):
    """
    평탄화된 키를 중첩된 딕셔너리로 업데이트하는 함수.
    """
    for key, value in flat_dict.items():
        # 평탄화된 키 분리, 예: "dataset.mode" -> ["dataset", "mode"]
        keys = key.split(".")
        sub_config = config
        for sub_key in keys[:-1]:  # 마지막 키 전까지 딕셔너리 탐색
            if sub_key not in sub_config:
                sub_config[sub_key] = {}  # 하위 딕셔너리가 없으면 생성
            sub_config = sub_config[sub_key]
        sub_config[keys[-1]] = value  # 마지막 키에 값 설정


def main():
    cfg = load_config()

    # ***********************************************************************************************************
    # ArgumentParser 객체 생성
    # utils.py에서 정의한 Parser 클래스 사용해 명령행 인자처리 "객체"
    name_data = cfg.get("name_data", "default_name")  # 기본값 설정 가능
    parser = argparse.ArgumentParser(description=f'Train the {name_data} classifier',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ***********************************************************************************************************
    # 명령행 인자 정의 parser.add_argument
    # bash에 python3 main.py --mode train --scope mnist --dir_log ./log
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="Test parser")

    # 중첩된 딕셔너리를 직접 처리
    for key, value in cfg.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parser.add_argument(
                    f"--{key}.{sub_key}", default=sub_value, type=type(sub_value))
        else:
            parser.add_argument(f"--{key}", default=value, type=type(value))

    # 3. 명령행 인자 파싱
    args = parser.parse_args()
    flat_args = vars(args)  # 평탄화된 명령행 인자 딕셔너리

    # 4. YAML 설정에 명령행 인자 병합
    nested_update(cfg, flat_args)  # 평탄화된 명령행 인자자

    PARSER = Parser(parser)
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    mode = vars(ARGS)["mode"]
    if mode == "train":
        # 데이터셋 로드
        ### ver2
        '''
        transform_train = transforms.Compose([Normalization(mean=[0.4567, 0.4431, 0.4086], std=[0.2680, 0.2649, 0.2806]),
                                    RandomFlip(), ToTensor()])
                                    '''
        transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5),
                                    RandomFlip(), ToTensor()])
        
        full_dataset = CustomedDataset_MultiClass_Seg(cfg, mode='train', transform=transform_train)
        loader_train, loader_val = buildDataLoader(full_dataset, cfg)
        trainer = Trainer(cfg=cfg)
        net, optim, loss_fn = build_settings_segmentation(cfg)
        train = trainer.train(loader_train, loader_val, net, optim, loss_fn)

    elif mode == 'test':
        ##  ver1, ver2.
        '''
        transform_test = transforms.Compose([Normalization(mean=[0.4567, 0.4431, 0.4086], std=[0.2680, 0.2649, 0.2806]), ToTensor()])
        '''
     
        transform_test = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

        test_dataset = CustomedDataset_MultiClass_Seg(cfg, mode='test', transform=transform_test)
        loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=False)

        trainer = Trainer(cfg=cfg)
        net, optim, loss_fn = build_settings_segmentation(cfg)
        test = trainer.test(loader_test, net, loss_fn)


if __name__ == '__main__':
    main()




# 유넷으로 classifier 말고 regression train.py 만 바꿔서 욱여넣어봐 최대한..
#TEST: AVERAGE LOSS: 2.664155
#TEST: AVERAGE ACC: 0.529627
#TEST: AVERAGE IOU: 0.158872