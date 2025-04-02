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

        gpu_id = cfg["created_model"]["device"] 
        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
        
        self.dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        self.dir_log = os.path.join(self.dir_log, self.scope, self.name_data, self.model_name)
        self.data_filepath = os.path.join(self.dir_data, self.name_data, self.mode)

        self.fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        self.fn_denorm = lambda x, mean, std: (x * std) + mean
        self.fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True).float() 
        self.logger = SummaryWriter(log_dir=os.path.join(self.dir_log, self.mode))
        self.logger_val = SummaryWriter(log_dir=os.path.join(self.dir_log, 'val'))

    def train_one_epoch(self, epoch, loader_train, net, optim_G, optim_D, loss_fn):
        num_batch_train = len(loader_train)
        
        netG_a2b = net['G_a2b']
        netG_b2a = net['G_b2a']
        netD_a = net['D_a']
        netD_b = net['D_b']
        
        loss_gan = loss_fn['gan']
        loss_cyc = loss_fn['cycle']
        loss_ident = loss_fn['identity']
        
        netG_a2b.train()
        netG_b2a.train()
        netD_a.train()
        netD_b.train()
        
        loss_D_train = []
        loss_D_a_train = []
        loss_D_b_train = []
        
        loss_G_train = []
        loss_G_gan_a2b_train = []
        loss_G_gan_b2a_train = []
        loss_G_gan_train = []
        loss_G_cyc_train = []
        loss_G_ident_train = []
        
        for i, batch in enumerate(loader_train, 1):
            
            input_a = batch['data_a'].to(self.device)
            input_b = batch['data_b'].to(self.device)
            
            # forward netG
            output_b = netG_a2b(input_a)
            output_a = netG_b2a(input_b)
            
            recon_a = netG_b2a(output_b)
            recon_b = netG_a2b(output_a)
            
            # -----------------------------------------------------
            # Train Discriminator
            # -----------------------------------------------------
            for net in [netD_a, netD_b]:
                for param in net.parameters():
                    param.required_grad = True
            
            optim_D.zero_grad()
            
            ## D_a
            pred_real_a = netD_a(input_a)              # torch.Size([4, 1, 8, 8])
            pred_fake_a = netD_a(output_a.detach())
            loss_D_a_real = loss_gan(pred_real_a, torch.ones_like(pred_real_a))
            loss_D_a_fake = loss_gan(pred_fake_a, torch.zeros_like(pred_fake_a))
            loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

            ## D_b
            pred_real_b = netD_b(input_b)
            pred_fake_b = netD_b(output_b.detach())
            loss_D_b_real = loss_gan(pred_real_b, torch.ones_like(pred_real_b))
            loss_D_b_fake = loss_gan(pred_fake_b, torch.zeros_like(pred_fake_b))
            loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)        
            
            ## Discriminator Loss ##
            loss_D = loss_D_a + loss_D_b
            loss_D.backward()
            optim_D.step()
            
            # -----------------------------------------------------
            # Train Generator
            # -----------------------------------------------------            
            for net in [netD_a, netD_b]:
                for param in net.parameters():
                    param.required_grad = False           
            
            optim_G.zero_grad()
            
            # GAN Loss
            pred_fake_a = netD_a(output_a)
            pred_fake_b = netD_b(output_b)   # torch.Size([4, 1, 8, 8])
            
            loss_G_a2b = loss_gan(pred_fake_a, torch.ones_like(pred_fake_a))
            loss_G_b2a = loss_gan(pred_fake_b, torch.ones_like(pred_fake_b))
            loss_G_gan = 0.5 * (loss_G_a2b + loss_G_b2a)
            
            # Cycle loss
            loss_cycle_a = loss_cyc(recon_a, input_a)
            loss_cycle_b = loss_cyc(recon_b, input_b)
            loss_G_cycle = 0.5 * (loss_cycle_a + loss_cycle_b)
            
            # identity loss 
            ident_a = netG_b2a(input_a)
            ident_b = netG_a2b(input_b)
            
            loss_ident_a = loss_ident(ident_a, input_a)
            loss_ident_b = loss_ident(ident_b, input_b)
            loss_G_ident = 0.5 * (loss_ident_a + loss_ident_b)
            
            # Total Loss
            loss_G = loss_G_gan + (float(self.lambda_cyc) * loss_G_cycle) + (float(self.lambda_id) * loss_G_ident)
            
            loss_G.backward()
            optim_G.step()

            result_dir_train_a2b = "./result/CycleGan_mse/train/a2b"
            result_dir_train_b2a = "./result/CycleGan_mse/train/b2a"
            os.makedirs(result_dir_train_a2b, exist_ok=True)
            os.makedirs(result_dir_train_b2a, exist_ok=True)

            if epoch % 5 == 0:  # 🔹 5 에폭마다 저장
                idx = -1
                # 1. Normalize & Convert to NumPy
                real_A_np = self.fn_tonumpy(self.fn_denorm(input_a[idx,:,:,:].unsqueeze(0), mean=0.5, std=0.5))   # input a 
                fake_B_np = self.fn_tonumpy(self.fn_denorm(output_b[idx,:,:,:].unsqueeze(0), mean=0.5, std=0.5))  # a2b output 
                recon_A_np = self.fn_tonumpy(self.fn_denorm(recon_a[idx,:,:,:].unsqueeze(0), mean=0.5, std=0.5))  # recon a 
                
                real_B_np = self.fn_tonumpy(self.fn_denorm(input_b[idx,:,:,:].unsqueeze(0), mean=0.5, std=0.5))   # input b
                fake_A_np = self.fn_tonumpy(self.fn_denorm(output_a[idx,:,:,:].unsqueeze(0), mean=0.5, std=0.5))  # b2a output
                recon_B_np = self.fn_tonumpy(self.fn_denorm(recon_b[idx,:,:,:].unsqueeze(0), mean=0.5, std=0.5))  # recon b

                # 2. Clipping & Tensor 변환 (HWC → CHW)
                real_A_np = np.clip(real_A_np, 0, 1)
                fake_B_np = np.clip(fake_B_np, 0, 1)
                recon_A_np = np.clip(recon_A_np, 0, 1)

                real_B_np = np.clip(real_B_np, 0, 1)
                fake_A_np = np.clip(fake_A_np, 0, 1)
                recon_B_np = np.clip(recon_B_np, 0, 1)


                # 3. 저장할 경로 설정
                id = num_batch_train * (epoch - 1) + i
                save_paths = {
                            "a2b": [
                                (real_A_np, f"{id}_1.png"),
                                (fake_B_np, f"{id}_1_o.png"),
                                (recon_A_np, f"{id}_1_r.png"),
                            ],
                            "b2a": [
                                (real_B_np, f"{id}_1.png"),
                                (fake_A_np, f"{id}_1_o.png"),
                                (recon_B_np, f"{id}_1_r.png"),
                            ]}
                
                for folder, images in save_paths.items():
                    save_dir = os.path.join(f"./result/CycleGan_mse/train/{folder}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    for img_np, filename in images:
                        img_path = os.path.join(save_dir, filename)
                        img_np = img_np.squeeze(0)  # 배치 차원 제거
                        plt.imsave(img_path, img_np)
                
                # 4. TensorBoard에 이미지 추가
                self.logger.add_image('Real_A', torch.from_numpy(real_A_np.squeeze(0)).permute(2, 0, 1), id)
                self.logger.add_image('Fake_B', torch.from_numpy(fake_B_np.squeeze(0)).permute(2, 0, 1), id)
                self.logger.add_image('Recon_A', torch.from_numpy(recon_A_np.squeeze(0)).permute(2, 0, 1), id)
                self.logger.add_image('Real_B', torch.from_numpy(real_B_np.squeeze(0)).permute(2, 0, 1), id)
                self.logger.add_image('Fake_A', torch.from_numpy(fake_A_np.squeeze(0)).permute(2, 0, 1), id)
                self.logger.add_image('Recon_B', torch.from_numpy(recon_B_np.squeeze(0)).permute(2, 0, 1), id)
 
            loss_D_train += [loss_D.item()]
            loss_D_a_train += [loss_D_a.item()]
            loss_D_b_train += [loss_D_b.item()]
            
            loss_G_train += [loss_G.item()]
            loss_G_gan_train += [loss_G_gan.item()]
            loss_G_gan_a2b_train += [loss_G_a2b.item()]
            loss_G_gan_b2a_train += [loss_G_b2a.item()]
            loss_G_cyc_train += [loss_G_cycle.item()]
            loss_G_ident_train += [loss_G_ident.item()]

            self.logger.add_scalar('loss_D_a', np.mean(loss_D_a_train), epoch)
            self.logger.add_scalar('loss_D_b', np.mean(loss_D_b_train), epoch)
            self.logger.add_scalar('loss_G_gan', np.mean(loss_G_gan_train), epoch)
            self.logger.add_scalar('loss_G_gan_a2b', np.mean(loss_G_gan_a2b_train), epoch)
            self.logger.add_scalar('loss_G_gan_b2a', np.mean(loss_G_gan_b2a_train), epoch)
            self.logger.add_scalar('loss_G_cyc', np.mean(loss_G_cyc_train), epoch)
            self.logger.add_scalar('loss_G_ident', np.mean(loss_G_ident_train), epoch)

            print("TRAIN : [Epoch %d/%d] [Batch %d/%d] [GEN: %.4f | DISC: %.4f ]"
                %  (epoch, self.cfg["num_epoch"], i, num_batch_train, 
                    np.mean(loss_G_train), np.mean(loss_D_train)))

        return loss_G_train, loss_D_train
    
    def validate(self, epoch, loader_val, net, loss_fn):
        netG_a2b = net['G_a2b']
        netG_b2a = net['G_b2a']
        netD_a = net['D_a']
        netD_b = net['D_b']
        
        loss_gan = loss_fn['gan']
        loss_cyc = loss_fn['cycle']
        loss_ident = loss_fn['identity']
        
        loss_G_val = []
        loss_G_gan_val = []
        loss_G_gan_a2b_val = []
        loss_G_gan_b2a_val = []
        loss_G_cyc_val = []
        loss_G_ident_val = []
        
        netG_a2b.eval()
        netG_b2a.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(loader_val, 1):
                num_batch_val = len(loader_val)
                input_a = batch['data_a'].to(self.device)
                input_b = batch['data_b'].to(self.device)
                
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)
                
                recon_a = netG_b2a(output_b)
                recon_b = netG_a2b(output_a)
                
                # GAN Loss
                pred_fake_a = netD_a(output_a)
                pred_fake_b = netD_b(output_b)
                
                loss_G_a2b = loss_gan(pred_fake_a, torch.ones_like(pred_fake_a))
                loss_G_b2a = loss_gan(pred_fake_b, torch.ones_like(pred_fake_b))
                loss_G_gan = 0.5 * (loss_G_a2b + loss_G_b2a)
                
                # Cycle loss
                loss_cycle_a = loss_cyc(recon_a, input_a)
                loss_cycle_b = loss_cyc(recon_b, input_b)
                loss_G_cycle = 0.5 * (loss_cycle_a + loss_cycle_b)
                
                # Identity loss
                ident_a = netG_b2a(input_a)
                ident_b = netG_a2b(input_b)
                
                loss_ident_a = loss_ident(ident_a, input_a)
                loss_ident_b = loss_ident(ident_b, input_b)
                loss_G_ident = 0.5 * (loss_ident_a + loss_ident_b)
                
                # Total Loss
                loss_G = loss_G_gan + (float(self.lambda_cyc) * loss_G_cycle) + (float(self.lambda_id) * loss_G_ident)
                
                loss_G_val.append(loss_G.item())
                loss_G_gan_val.append(loss_G_gan.item())
                loss_G_cyc_val.append(loss_G_cycle.item())
                loss_G_ident_val.append(loss_G_ident.item())
                
                loss_G_gan_a2b_val.append(loss_G_a2b.item())
                loss_G_gan_b2a_val.append(loss_G_b2a.item())
                
        self.logger_val.add_scalar('loss_G', np.mean(loss_G_val), epoch)
        self.logger_val.add_scalar('loss_G_gan', np.mean(loss_G_gan_val), epoch)
        self.logger_val.add_scalar('loss_G_gan_a2b', np.mean(loss_G_gan_a2b_val), epoch)
        self.logger_val.add_scalar('loss_G_gan_b2a', np.mean(loss_G_gan_b2a_val), epoch)
        self.logger_val.add_scalar('loss_G_cyc', np.mean(loss_G_cyc_val), epoch)
        self.logger_val.add_scalar('loss_G_ident', np.mean(loss_G_ident_val), epoch)
        
        print("VALID : [Epoch %d/%d] [Batch %d/%d] [GEN: %.4f]"
%  (epoch, self.cfg["num_epoch"], i, num_batch_val, np.mean(loss_G_val)))

    
    def train(self, loader_train, loader_val, net, optim_G, optim_D, loss_fn):
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)
        # setup tensorboard
        st_epoch = 0
        
        if self.train_continue == 'True' :
            print("Continuing from the last checkpoint...")
            net, optim_G, optim_D, loaded_epoch = self.load(dir_chck, net,
                                                            optim_G=optim_G, optim_D=optim_D, 
                                                            mode='train')
            st_epoch = loaded_epoch
            
        else:
            print("Starting training from scratch...")
            
        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            loss_G_train, loss_D_train = self.train_one_epoch(epoch, loader_train, 
                                            net, optim_G, optim_D, loss_fn)
            
            self.logger.add_scalar('loss_G', np.mean(loss_G_train), epoch)
            self.logger.add_scalar('loss_D', np.mean(loss_D_train), epoch)
        
            self.validate(epoch, loader_val, net, loss_fn)

            # save
            if (epoch % self.num_freq_save) == 0:
                self.save(dir_chck, net, optim_G, optim_D, epoch)
                

        self.logger.close()

    def test(self, loader_test_a, loader_test_b, net, loss_fn):

        # setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name)

        netG_a2b = net['G_a2b']
        netG_b2a = net['G_b2a']

        # load from checkpoints
        net, loaded_epoch = self.load(dir_chck = dir_chck, net=net, optim_G=[], optim_D= []) # map
        # net.to(device)
        print('Epoch :', loaded_epoch)

        save_output_dir_a2b = "./result/CycleGan_mse/test/a2b"
        save_output_dir_b2a = "./result/CycleGan_mse/test/b2a"
        os.makedirs(save_output_dir_a2b, exist_ok=True)
        os.makedirs(save_output_dir_b2a, exist_ok=True)

        # test phase
        with torch.no_grad():
            netG_a2b.eval()
            netG_b2a.eval()

        for i, batch_a in enumerate(loader_test_a, 1):
            input_a = batch_a['data_a'].to(self.device)
            fake_b = netG_a2b(input_a)

            # 🔹 1. Normalize & Convert to NumPy
            real_A_np = self.fn_tonumpy(self.fn_denorm(input_a, mean=0.5, std=0.5)) 
            fake_B_np = self.fn_tonumpy(self.fn_denorm(fake_b, mean=0.5, std=0.5))

            # 🔹 2. Clipping & Tensor 변환
            real_A_tensor = torch.from_numpy(np.clip(real_A_np, 0, 1)).permute(0, 3, 1, 2) 
            fake_B_tensor = torch.from_numpy(np.clip(fake_B_np, 0, 1)).permute(0, 3, 1, 2)

            # 🔹 3. 한 장의 이미지로 저장 (A → B)
            image_a2b = torch.cat((real_A_tensor, fake_B_tensor), dim=3)  # 세로로 연결
            save_path_a2b = os.path.join(save_output_dir_a2b, f'{i:05d}_A2B.png')
            plt.imsave(save_path_a2b, image_a2b.squeeze(0).permute(1, 2, 0).cpu().numpy())

        for i, batch_b in enumerate(loader_test_b, 1):
            input_b = batch_b['data_b'].to(self.device)
            fake_a = netG_b2a(input_b)

            # 🔹 1. Normalize & Convert to NumPy
            real_B_np = self.fn_tonumpy(self.fn_denorm(input_b, mean=0.5, std=0.5))
            fake_A_np = self.fn_tonumpy(self.fn_denorm(fake_a, mean=0.5, std=0.5))

            # 🔹 2. Clipping & Tensor 변환
            real_B_tensor = torch.from_numpy(np.clip(real_B_np, 0, 1)).permute(0, 3, 1, 2)
            fake_A_tensor = torch.from_numpy(np.clip(fake_A_np, 0, 1)).permute(0, 3, 1, 2)

            # 🔹 3. 한 장의 이미지로 저장 (B → A)
            image_b2a = torch.cat((real_B_tensor, fake_A_tensor), dim=3)  # 세로로 연결
            save_path_b2a = os.path.join(save_output_dir_b2a, f'{i:05d}_B2A.png')
            plt.imsave(save_path_b2a, image_b2a.squeeze(0).permute(1, 2, 0).cpu().numpy())

        print(f"Test images saved in {save_output_dir_a2b} and {save_output_dir_b2a}")


    def load(self, dir_chck, net, optim_G, optim_D, epoch=[], mode='test'):
        
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        netG_a2b = net['G_a2b']
        netG_b2a = net['G_b2a']
        netD_a = net['D_a']
        netD_b = net['D_b']
        
        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch)) #, map_location=
        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])
            netD_a.load_state_dict(dict_net['netD_a'])
            netD_b.load_state_dict(dict_net['netD_b'])
            
            optim_G.load_state_dict(dict_net['optim_G'])
            optim_D.load_state_dict(dict_net['optim_D'])

            return net, optim_G, optim_D, epoch

        elif mode == 'test':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])
            netD_a.load_state_dict(dict_net['netD_a'])
            netD_b.load_state_dict(dict_net['netD_b'])

            return net, epoch

    def save(self, dir_chck, net, optim_G, optim_D, epoch):
        if not os.path.exists(self.dir_chck):
            os.makedirs(self.dir_chck)
        
        netG_a2b = net['G_a2b']
        netG_b2a = net['G_b2a']
        netD_a = net['D_a']
        netD_b = net['D_b']
        
        torch.save({'netG_a2b': netG_a2b.state_dict(),
                    'netG_b2a': netG_b2a.state_dict(),
                    'netD_a': netD_a.state_dict(),
                    'netD_b': netD_b.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict()},
                    '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    