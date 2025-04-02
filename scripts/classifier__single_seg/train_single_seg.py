import os
import torch
import numpy as np 

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


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
    
        self.device =torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        self.dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        self.dir_log = os.path.join(self.dir_log, self.scope, self.name_data, self.model_name)
        self.data_filepath = os.path.join(self.dir_data, self.name_data, self.mode)
        
        self.fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        self.fn_denorm = lambda x, mean, std: (x * std) + mean
        # self.fn_class = lambda x: 1.0 * (x > 0.5)                             # BCEwithlogitLoss
        self.fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True).float()  # softmax(2) -> cross entropyloss

    
    
    def train_one_epoch(self, epoch, loader_train, net, optim, loss_fn, writer_train):
        num_batch_train = len(loader_train)
        
        net.train()
        loss_train = []

        for i, batch in enumerate(loader_train, 1):
            def should(freq):
                return freq > 0 and (i % freq == 0 or i == num_batch_train)

            input = batch['image'].to(self.device) 
            label = batch['label'].to(self.device)

            output = net(input)
            # output =  F.log_softmax(output, dim=1)  # NLL loss - log softmax이 입력값이어야

            # backward netD
            optim.zero_grad()

            #loss = loss_fn(output, label)  # BCEwithLogitLoss
            loss = loss_fn(output, label.squeeze(1).long()) # softmax(2), NLL LOSS
            
            loss.backward()
            optim.step()

            # get losses
            loss_train += [loss.item()]

            print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f' %(epoch, i, num_batch_train, np.mean(loss_train)))
            
            label_np = self.fn_tonumpy(label)
            input_np = self.fn_tonumpy(self.fn_denorm(input, mean=0.5, std=0.5))
            output_np = self.fn_tonumpy(self.fn_class(output))
 
            writer_train.add_image('label', label_np, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            writer_train.add_image('input', input_np, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            writer_train.add_image('output', output_np, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            
        return np.mean(loss_train)
    
    def val_one_batch(self, batch, net, loss_fn):

        # Set model to evaluation mode
        net.eval()

        # Move data to the appropriate device
        input = batch['image'].to(self.device)
        label = batch['label'].to(self.device)

        with torch.no_grad(): 
            output = net(input)
            # Calculate loss
            #loss = loss_fn(output, label)
            loss = loss_fn(output, label.squeeze(1).long())


        return loss.item(), output

        
    def train(self, loader_train, loader_val, net, optim, loss_fn):
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(self.dir_log,'train'))
        writer_val = SummaryWriter(log_dir=os.path.join(self.dir_log,'val'))
        
        st_epoch = 0
        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            
            loss_train = self.train_one_epoch(epoch, loader_train, net, optim, loss_fn, writer_train)
            
            writer_train.add_scalar('loss', np.mean(loss_train), epoch)
            
            with torch.no_grad():
                num_batch_val = len(loader_val)
                net.eval()
                loss_val = []
                
                for i, batch in enumerate(loader_val, 1):
                    loss, output = self.val_one_batch(batch, net, loss_fn)
                    
                    loss_val.append(loss)

                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f' %(epoch, i, num_batch_val, np.mean(loss_val)))
                    
                    label_np = self.fn_tonumpy(batch['label'])
                    input_np = self.fn_tonumpy(self.fn_denorm(batch['image'], mean=0.5, std=0.5))
                    output_np = self.fn_tonumpy(self.fn_class(output))

                    writer_val.add_image('label', label_np, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                    writer_val.add_image('input', input_np, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                    writer_val.add_image('output', output_np, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                    
                    
            writer_val.add_scalar('loss', np.mean(loss_val), epoch)
       
            
            ## save
            #if (epoch % self.num_freq_save) == 0:
            self.save(dir_chck, net, optim, epoch)

        writer_train.close()
        writer_val.close()
    
    
    def load(self, dir_chck, net, optim=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            net.load_state_dict(dict_net['net'])
            optim.load_state_dict(dict_net['optim'])

            return net, optim, epoch

        elif mode == 'test':
            net.load_state_dict(dict_net['net'])

            return net, epoch
    
    
    def save(self, dir_chck, net, optim, epoch):
        if not os.path.exists(self.dir_chck):
            os.makedirs(self.dir_chck)

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                '%s/model_epoch%04d.pth' % (dir_chck, epoch))        
    
    
    def test(self, loader_test, net ,loss_fn):
        
        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        num_batch_test = len(loader_test)

        ## load from checkpoints
        net, loaded_epoch = self.load(dir_chck, net, mode=self.mode)
        print('Epoch :', loaded_epoch)
        
        ## test phase
        with torch.no_grad():
            net.eval()

            loss_test = []

            for i, batch in enumerate(loader_test, 1):
                loss, output = self.val_one_batch(batch, net, loss_fn)
                    
                loss_test.append(loss)

                print('TEST: BATCH %04d/%04d: LOSS: %.4f' % (i, num_batch_test, np.mean(loss_test)))

            print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_test)))
   

