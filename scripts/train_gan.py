import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision.utils as vutils


class Trainer:
    def __init__(self, cfg):

        self.name_data = cfg["name_data"]
        self.dir_data = cfg["dir_data"]
        self.model_name = cfg["model_name"]
        self.mode = cfg["mode"]
        self.cfg = cfg["train"]

        self.dir_checkpoint = self.cfg["dir_checkpoint"]

        for key, value in self.cfg.items():
            setattr(self, key, value)

        gpu_id = cfg["created_model"]["gpu_id"] 
        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
        
        self.dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        self.dir_log = os.path.join(self.dir_log, self.scope, self.name_data, self.model_name)
        self.data_filepath = os.path.join(self.dir_data, self.name_data, self.mode)

        self.fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        self.fn_denorm = lambda x, mean, std: (x * std) + mean
        self.fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True).float() 

    def train_one_epoch(self, epoch, loader_train, net_G, net_D, optim_G, optim_D, loss_fn, writer_train):
        num_batch_train = len(loader_train)
        
        net_G.train()
        net_D.train()
        
        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        for i, batch in enumerate(loader_train, 1):
            
            label = batch['label'].to(self.device)                          # torch.Size([64, 1, 32, 32])
            # Sample noise as generator input
            input = torch.randn(label.shape[0], 100, 1,1).to(self.device)   # torch.Size([64, 100, 1, 1])
            
            output = net_G(input)                                           # torch.Size([64, 1, 32, 32])
            # ---------------------------
            #  Train Discriminator
            # ---------------------------
            for param in net_D.parameters():
                param.requires_grad = True
            optim_D.zero_grad()
            
            pred_real = net_D(label)            # torch.Size([64, 1, 1, 1])
            pred_fake = net_D(output.detach()) 
            
            loss_D_real = loss_fn(pred_real, torch.ones_like(pred_real))
            loss_D_fake = loss_fn(pred_fake, torch.zeros_like(pred_real))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            
            loss_D.backward()
            optim_D.step()
            # ---------------------------
            ## G에서의 backpropagation
            # ---------------------------
            for param in net_D.parameters():
                param.requires_grad = False
                
            optim_G.zero_grad()
            
            pred_fake = net_D(output)
            loss_G = loss_fn(pred_fake, torch.ones_like(pred_fake))
            
            loss_G.backward()
            optim_G.step()

            loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]
            
            print("[Epoch %d/%d] [Batch %d/%d] [GEN: %.4f | DISC REAL: %.4f | DISC FAKE: %.4f]"
            % (epoch, self.cfg["num_epoch"], i, num_batch_train, 
               np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))
            
            result_dir_train = "./result/gan/train"
            os.makedirs(result_dir_train, exist_ok=True)

            if i % 300 == 0:
                output_numpy = self.fn_tonumpy(self.fn_denorm(output, mean=0.5, std=0.5))  # (64, 32, 32, 1)
                output = np.clip(output_numpy, a_min=0, a_max=1)
                
                output = output.transpose(0,3,1,2)  # (B, C, H, W) → (64, 1, 32, 32)
                output_tensor = torch.from_numpy(output)

                # 🔥 8x8 형태로 정렬하여 한 장의 이미지로 만들기
                grid_image = vutils.make_grid(output_tensor, nrow=8, normalize=True, padding=2)  # (C, H, W)

                id = num_batch_train * (epoch - 1) + i
                image_dir = os.path.join(result_dir_train, 'image_cifar-10')
                os.makedirs(image_dir, exist_ok=True)  
                
                save_path = os.path.join(result_dir_train, 'image_cifar-10', '%09d_output.png' % id)
                grid_np = grid_image.permute(1, 2, 0).cpu().numpy()
                plt.imsave(save_path, grid_np, cmap=None)
                
                writer_train.add_image('output_grid', grid_image, id)
                
        return loss_G_train, loss_D_real_train, loss_D_fake_train

    
    def train(self, loader_train, loader_val, net_G, net_D, optim_G, optim_D, loss_fn):
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        # setup tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(self.dir_log, 'train'))

        st_epoch = 0
        
        if self.train_continue == 'True' :
            print("Continuing from the last checkpoint...")
            net_G, net_D, optim_G, optim_D, loaded_epoch = self.load(dir_chck, 
                                                                    net_G=net_G, net_D=net_D,
                                                                    optim_G=optim_G, optim_D=optim_D, 
                                                                    mode='train')
            st_epoch = loaded_epoch
            
        else:
            print("Starting training from scratch...")

        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            loss_G_train, loss_D_real_train, loss_D_fake_train = self.train_one_epoch(epoch, loader_train, 
                                            net_G, net_D, optim_G, optim_D, loss_fn, writer_train)
            
            writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

            # save
            if (epoch % self.num_freq_save) == 0:
                self.save(dir_chck, net_G, net_D, optim_G, optim_D, epoch)

        writer_train.close()

    def test(self, loader_test, net_G, net_D, loss_fn):

        # setup dataset
        dir_chck = os.path.join(
            self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        num_batch_test = len(loader_test)

        # load from checkpoints
        net_G, net_D, optim_G, optim_D, loaded_epoch = self.load(dir_chck = dir_chck, 
                                                                net_G =net_G, net_D=net_D, 
                                                                optim_G=optim_G, optim_D= optim_D)
        print('Epoch :', loaded_epoch)


        save_output_dir = "./result/gan/test"
        os.makedirs(save_output_dir, exist_ok=True)

        # test phase
        with torch.no_grad():
            net_G.eval()
            # net_D.eval() 이건 안필요

            input = torch.randn(self.batch_size, 100).to(self.device)
            output = net_G(input)
            
            output = self.fn_tonumpy(self.fn_denorm(output, mean=0.5, std=0.5))

            for j in range(output.shape[0]):
                output_ = output[j]
                
                output_ = np.clip(output_, a_min=0, a_max=1)
                plt.imsave()













    def load(self, dir_chck, net_G, net_D, 
            optim_G, optim_D, epoch=[], mode='train'):
        
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            net_G.load_state_dict(dict_net['net_G'])
            net_D.load_state_dict(dict_net['net_D'])
            
            optim_G.load_state_dict(dict_net['optim_G'])
            optim_D.load_state_dict(dict_net['optim_D'])

            return net_G, net_D, optim_G, optim_D, epoch

        elif mode == 'test':
            net_G.load_state_dict(dict_net['net_G'])
            net_D.load_state_dict(dict_net['net_D'])

            return net_G, net_D, epoch

    def save(self, dir_chck, net_G, net_D, optim_G, optim_D, epoch):
        if not os.path.exists(self.dir_chck):
            os.makedirs(self.dir_chck)

        torch.save({'net_G': net_G.state_dict(), 'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),'optim_D': optim_D.state_dict()},
                    '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    