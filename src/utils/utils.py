import os
import logging
import torch
import argparse
import yaml
from src.models.created_model import MODEL_CLASSES
import torch.nn as nn
import torch.nn.functional as F

import itertools

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

        log_dir = os.path.join(params_dict['train.dir_log'])
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
    net = model_class(cfg["nch_in"], cfg["nch_out"]).to(device)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    params = net.parameters()
    optim = torch.optim.Adam(params, lr = cfg["lr"])
    print(net)
    
    return net, optim, loss_fn

def build_settings_segmentation(cfg):
    cfg = cfg["created_model"]
    gpu_id = 3

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    model_class = MODEL_CLASSES[cfg["model_name"]]
    net = model_class().to(device)
    
    ########################### isbi ###############################
    '''
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    '''
    
    #### ver2. output channel 21 -> softmax 적용하지 않음. CE에서 나부적으로 softmax 적용됨
    #loss_fn = nn.CrossEntropyLoss(ignore_index = 255).to(device)
    loss_fn = CombinedLoss(weight=0.1)


    #### ver1, ver3. output channel 21 -> softmax 적용하지 않음.
    '''
    loss_fn = nn.NLLLoss().to(device)
    '''

    params = net.parameters()
    optim = torch.optim.Adam(params, lr = cfg["lr"])
    
    #print(net)
    return net, optim, loss_fn




class CombinedLoss(nn.Module):
    def __init__(self, weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.device = torch.device(f"cuda:{3}") if torch.cuda.is_available() else torch.device("cpu")
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index = 255).to(self.device)
        
    def dice_loss(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        
        targets_one_hot  = targets.clone().squeeze(1)
        targets_one_hot [targets_one_hot  == 255] = 0
        targets_one_hot  = F.one_hot(targets_one_hot, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
    
        intersection = (inputs * targets_one_hot).sum((2, 3))
        dice_score = (2. * intersection + self.smooth) / (inputs.sum((2, 3)) + targets_one_hot.sum((2, 3)) + self.smooth)
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss
    
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy_loss(inputs, targets.squeeze(1))
        dl_loss = self.dice_loss(inputs, targets)

        return self.weight * ce_loss + (1 - self.weight) * dl_loss
    


def build_settings_gan(cfg):
    cfg = cfg["created_model"]
    gpu_id = 3

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    # Generator와 Discriminator 인자 설정
    in_channels_G = 100  # Noise vector 크기 (DCGAN 논문 기준)
    out_channels_G = cfg["nch_out"]  # 생성할 이미지 채널 수 (ex. 흑백 1, 컬러 3)
    in_channels_D = cfg["nch_in"]  # Discriminator 입력 이미지 채널 수
    out_channels_D = 1  # Real/Fake 구분이므로 출력은 1
    nker = cfg["nker"]  # 필터 개수 (기본값: 64)
    norm = "bnorm"  # 기본 정규화 방식
    
    model_G = MODEL_CLASSES["Generator"]
    net_G = model_G(in_channels=in_channels_G, out_channels=out_channels_G, nker=nker, norm=norm).to(device)
    model_D = MODEL_CLASSES["Discriminator"]
    net_D = model_D(in_channels=in_channels_D, out_channels=out_channels_D, nker=nker, norm=norm).to(device)
    
    init_weights(net_G, init_type='normal', init_gain=0.02)
    init_weights(net_D, init_type='normal', init_gain=0.02)
    
    loss_fn = torch.nn.BCELoss()

    params_G = net_G.parameters()
    params_D = net_D.parameters()
    
    optim_G = torch.optim.Adam(params_G, lr = cfg["lr"], betas=[0.5, 0.999])
    optim_D = torch.optim.Adam(params_D, lr = cfg["lr"], betas=[0.5, 0.999])

    return net_G, net_D, optim_G, optim_D, loss_fn

## 네트워크 weights 초기화 하기
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
    
    
def build_settings_cyclegan(cfg):
    cfg = cfg["created_model"]
    gpu_id = cfg["device"]

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    # Generator와 Discriminator 인자 설정
    in_channels_G = 3 #100  Noise vector 크기 (DCGAN 논문 기준)
    out_channels_G = cfg["nch_out"]  # 생성할 이미지 채널 수 (ex. 흑백 1, 컬러 3)
    in_channels_D = cfg["nch_in"]  # Discriminator 입력 이미지 채널 수
    out_channels_D = 1  # Real/Fake 구분이므로 출력은 1
    nker = cfg["nker"]  # 필터 개수 (기본값: 64)
    norm = "inorm"  # 기본 정규화 방식
    
    model_G = MODEL_CLASSES["CycleGan"]
    netG_a2b = model_G(in_channels=in_channels_G, out_channels=out_channels_G, nker=nker, norm=norm, nblk=9).to(device)
    netG_b2a = model_G(in_channels=in_channels_G, out_channels=out_channels_G, nker=nker, norm=norm, nblk=9).to(device)
    
    model_D = MODEL_CLASSES["Discriminator"]
    netD_a = model_D(in_channels=in_channels_D, out_channels=out_channels_D, nker=nker, norm=norm).to(device)
    netD_b = model_D(in_channels=in_channels_D, out_channels=out_channels_D, nker=nker, norm=norm).to(device)
    
    init_weights(netG_a2b, init_type='normal', init_gain=0.02)
    init_weights(netG_b2a, init_type='normal', init_gain=0.02)
    init_weights(netD_a, init_type='normal', init_gain=0.02)
    init_weights(netD_b, init_type='normal', init_gain=0.02)
    
    net =  {'G_a2b': netG_a2b,
            'G_b2a': netG_b2a,
            'D_a': netD_a,
            'D_b': netD_b}
    
    # 손실함수 정의 
    #loss_fn_gan = nn.BCELoss()
    loss_fn_gan = nn.MSELoss()
    loss_fn_cycle = nn.L1Loss()
    loss_fn_identity = nn.L1Loss()
    loss_fn =  {'gan': loss_fn_gan,
                'cycle': loss_fn_cycle,
                'identity':loss_fn_identity}

    paramsG_a2b = netG_a2b.parameters()
    paramsG_b2a = netG_b2a.parameters()
    paramsD_a = netD_a.parameters()
    paramsD_b = netD_b.parameters()
    
    optim_G = torch.optim.Adam(itertools.chain(paramsG_a2b, paramsG_b2a), lr = cfg["lr"], betas=[0.5, 0.999])
    optim_D = torch.optim.Adam(itertools.chain(paramsD_a, paramsD_b), lr = cfg["lr"], betas=[0.5, 0.999])

    return net, optim_G, optim_D, loss_fn

