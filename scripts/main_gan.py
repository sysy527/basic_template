import sys
import os
import autorootcwd
import yaml
import argparse
import torch

from data.dataset import CustomedDataset, buildDataLoader
from data.dataset import Normalization, ToTensor
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from src.models.created_model import MODEL_CLASSES
from src.utils.utils import load_config, Parser
from scripts.train_gan import Trainer
from src.utils.utils import build_settings_gan


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
        transform_train = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                              transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        full_dataset = CustomedDataset(cfg, mode='train', transform=transform_train)
        loader_train, loader_val = buildDataLoader(full_dataset, cfg)
        trainer = Trainer(cfg=cfg)
        net_G, net_D, optim_G, optim_D, loss_fn = build_settings_gan(cfg)
        train = trainer.train(loader_train, loader_val, net_G, net_D, 
                                                        optim_G, optim_D, loss_fn)

    elif mode == 'test':

        transform_test = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5), transforms.ToTensor()])

        test_dataset = CustomedDataset(cfg, mode='test', transform=transform_test)
        loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False)

        trainer = Trainer(cfg=cfg)
        net_G, net_D, optim_G, optim_D, loss_fn = build_settings_gan(cfg)
        test = trainer.test(loader_test, net_G, net_D, loss_fn)


if __name__ == '__main__':
    main()


# cifar-10 3 channel - Normalize 3채널 / nker 64 / (32,32) / default yaml 파일에서 nch_in, out 3으로
# mnist 1 channel - Normalize 1채널[0.5],[0.5] / nker 64 / (32,32) / default yaml 파일에서 nch_in, out 1으로
