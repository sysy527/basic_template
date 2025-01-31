import os
import logging
import torch
import argparse
import yaml
from src.models.created_model import MODEL_CLASSES
import torch.nn as nn

class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()
                
    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['train.dir_log'], params_dict['train.scope'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)
        
        
def load_config(config_path="cfg/default.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)   # json으로 바꾸면 cfg["created_model"] 대신 cfg. 으로 쓸수있어어
    return config

def build_settings(cfg):
    cfg = cfg["created_model"]
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    
    model_class = MODEL_CLASSES[cfg["model_name"]]
    net = model_class().to(device)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    params = net.parameters()
    optim = torch.optim.Adam(params, lr = cfg["lr"])
    
    return net, optim, loss_fn



