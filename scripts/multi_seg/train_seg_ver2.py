import os

import matplotlib.pyplot as plt
import numpy as np
import torch
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
        self.batch_size = self.cfg["batch_size"]
        self.train_continue = self.cfg["train_continue"]

        for key, value in self.cfg.items():
            setattr(self, key, value)

        gpu_id = 3

        self.device = torch.device(
            f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
        self.dir_chck = os.path.join(
            self.dir_checkpoint, self.scope, self.name_data, self.model_name, 'loss_dl_0.5')
        self.dir_log = os.path.join(
            self.dir_log, self.scope, self.name_data, self.model_name, 'loss_dl_0.5')
        self.data_filepath = os.path.join(
            self.dir_data, self.name_data, self.mode)

        self.fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        self.fn_denorm = lambda x, mean, std: (x * torch.tensor(std, device=x.device).view(-1, 1, 1)) + torch.tensor(mean, device=x.device).view(-1, 1, 1)
        self.fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True).float() 


    def train_one_epoch(self, epoch, loader_train, net, optim, loss_fn, writer_train):
        num_batch_train = len(loader_train)

        net.train()
        loss_train = []

        for i, batch in enumerate(loader_train, 1):
            def should(freq):
                return freq > 0 and (i % freq == 0 or i == num_batch_train)
            
            # pascal
            input = batch['image'].to(self.device).type(torch.float32)   # torch.Size([32, 3, 96, 96])
            label = batch['label'].to(self.device).type(torch.long)   # torch.Size([32, 1, 96, 96])

            # randomcrop 시, "0" 이 50프로 넘는거만 학습에 넣기

            # ver2. index 그대로 / output(1) channel / Cross entropy loss
            output = net(input)           # torch.Size([32, 21, 96, 96])

            # backward netD
            optim.zero_grad()
            loss = loss_fn(output, label)

            # get losses
            loss.backward()
            optim.step()
            loss_train += [loss.item()]

            print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f' %
                    (epoch, i, num_batch_train, np.mean(loss_train)))

            label_np =  self.fn_tonumpy(label)
            input_np =  self.fn_tonumpy(
                        self.fn_denorm(input, mean=0.5, std=0.5))
            output_np = self.fn_tonumpy(self.fn_class(output))

        return np.mean(loss_train), {"input": input_np, "label": label_np, "output": output_np}

    def val_one_batch_sliding(self, batch, net, loss_fn):

        # Set model to evaluation mode
        net.eval()

        input = batch['image'].to(self.device).type(torch.float32)   # torch.Size([32, 3, 512, 512])
        label = batch['label'].to(self.device).type(torch.long)      # torch.Size([32, 1, 512, 512])

        patch_size = 96
        overlap = 0.5
        stride = int(patch_size * (1-overlap))

        _, _, H, W = input.shape
        output_full = torch.zeros((input.shape[0], 21, H, W)).to(self.device)       # num_classes = 21
        count_map = torch.zeros((input.shape[0], 1, H, W)).to(self.device)

        with torch.no_grad():
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    # patch 추출
                    patch_input = input[:, :, y:y +patch_size, x:x + patch_size] # torch.Size([32, 3, 96, 96])

                    # 모델 예측
                    patch_output = net(patch_input)                              # torch.Size([32, 21, 96, 96])
                    
                    # 예측 결과 누적
                    output_full[:, :, y:y + patch_size,x:x + patch_size] += patch_output
                    count_map[:, :, y:y + patch_size, x:x + patch_size] += 1

                #patch_output[W - patch_size: W+1]  # 테두리
            # 중첩된 부분의 평균

            # torch.Size([32, 21, 512, 512])
            output_full = output_full / (count_map.type(torch.float32) + 1e-6)
            # 최종 output 업데이트, not for loss

            # Calculate loss
            loss = loss_fn(output_full, label)
            output = output_full
            output = F.softmax(output)

        label_np =  self.fn_tonumpy(label)
        input_np =  self.fn_tonumpy(
                    self.fn_denorm(input, mean=0.5, std=0.5))
    
        output_np = self.fn_tonumpy(output)

        return loss.item(),  {"input": input_np, "label": label_np, "output": output_np}
    
    def val_one_batch(self, batch, net, loss_fn):

        # Set model to evaluation mode
        net.eval()

        input = batch['image'].to(self.device).type(torch.float32)   # torch.Size([32, 3, 512, 512])
        label = batch['label'].to(self.device).type(torch.long)      # torch.Size([32, 1, 512, 512])

        with torch.no_grad():
            output = net(input)                              # torch.Size([32, 21, 96, 96])

            # Calculate loss
            loss = loss_fn(output, label)
            output = F.softmax(output)

            label_np =  self.fn_tonumpy(label)
            input_np =  self.fn_tonumpy(
                        self.fn_denorm(input, mean=0.5, std=0.5))
        
            output_np = self.fn_tonumpy(output)

        return loss.item(),  {"input": input_np, "label": label_np, "output": output_np}

    def train(self, loader_train, loader_val, net, optim, loss_fn):
        dir_chck = os.path.join(
            self.dir_checkpoint, self.scope, self.name_data, self.model_name, 'loss_dl_0.5')
        # setup tensorboard
        writer_train = SummaryWriter(
            log_dir=os.path.join(self.dir_log, 'train'), comment="train")
        writer_val = SummaryWriter(log_dir=os.path.join(
            self.dir_log, 'val'), comment="val")

        st_epoch = 0

        if self.train_continue == 'True' :
            print("Continuing from the last checkpoint...")
            net, optim, loaded_epoch = self.load(dir_chck, net, optim, mode='train')
            st_epoch = loaded_epoch
        else:
            print("Starting training from scratch...")

        for epoch in range(st_epoch + 1, self.num_epoch + 1):

            loss_train, output = self.train_one_epoch(
                epoch, loader_train, net, optim, loss_fn, writer_train)
            
            writer_train.add_scalar('loss', np.mean(loss_train), epoch)
            writer_train.add_image(
                'input', output['input'], epoch, dataformats='NHWC')
            writer_train.add_image(
                'label', output['label'], epoch, dataformats='NHWC')
            writer_train.add_image(
                'output', output['output'], epoch, dataformats='NHWC')

            save_input_train_dir = "./result/pre/loss_dl_0.5/train/input"
            save_output_train_dir = "./result/pre/loss_dl_0.5/train/output"
            save_target_train_dir = "./result/pre/loss_dl_0.5/train/target"

            os.makedirs(save_input_train_dir, exist_ok=True)
            os.makedirs(save_output_train_dir, exist_ok=True)
            os.makedirs(save_target_train_dir, exist_ok=True)  

            with torch.no_grad():
                num_batch_val = len(loader_val)
                loss_val = []

                for i, batch in enumerate(loader_val, 1):
                    loss, output = self.val_one_batch(batch, net, loss_fn)

                    loss_val.append(loss)

                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f' %
                          (epoch, i, num_batch_val, np.mean(loss_val)))

                    input_np = output['input']
                    label_np = output['label']     # (32, 512, 512, 1)
                    output_np = output['output'] 

                    img_input = (input_np[-1] * 255).astype(np.uint8)
                    # (512, 512, 1)
                    img_label = label_np[-1]
                    # (512, 512, 21)
                    img_output = output_np[-1]


                    input_path = os.path.join(
                        save_input_train_dir, f"test_{i}.npy")
                    output_path = os.path.join(
                        save_output_train_dir, f"test_{i}.npy")
                    target_path = os.path.join(
                        save_target_train_dir, f"test_{i}.npy")
                    
                    # 3. 그림 저장
                    np.save(input_path, img_input)      # (512, 512, 3)
                    np.save(output_path, img_output)    # (512, 512, 21)
                    np.save(target_path, img_label)     # (512, 512, 1)

                avg_val_loss = np.mean(loss_val)
                writer_val.add_scalar('loss', avg_val_loss, epoch)

            # save
            if (epoch % self.num_freq_save) == 0:
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

    def test(self, loader_test, net, loss_fn):

        # setup dataset
        dir_chck = os.path.join(
            self.dir_checkpoint, self.scope, self.name_data, self.model_name, 'loss_dl_0.5')
        num_batch_test = len(loader_test)

        # load from checkpoints
        net, loaded_epoch = self.load(dir_chck, net, mode=self.mode)
        print('Epoch :', loaded_epoch)

        save_input_dir = "./result/pre/loss_dl_0.5/test/epoch300/input"
        save_output_dir = "./result/pre/loss_dl_0.5/test/epoch300/output"
        save_target_dir = "./result/pre/loss_dl_0.5/test/epoch300/target"

        os.makedirs(save_input_dir, exist_ok=True)
        os.makedirs(save_output_dir, exist_ok=True)
        os.makedirs(save_target_dir, exist_ok=True)  # 저장 폴더 생성

        # test phase
        with torch.no_grad():
            net.eval()

            loss_test = []
            acc_test = []
            iou_test = []
            global_idx = 0

            for i, batch in enumerate(loader_test, 1):
                # pascal

                input = batch['image'].to(self.device).type(torch.float32)  # 입력 이미지 (batch_size, 3, 512, 512)
                label = batch['label'].to(self.device).type(torch.long)     # 정답 (batch_size, 21, 512, 512)

                loss, output = self.val_one_batch(batch, net, loss_fn)
                loss_test.append(loss)

                print('TEST: BATCH %04d/%04d: LOSS: %.4f' %
                      (i, num_batch_test, np.mean(loss_test)))

                input_np = output['input']
                label_np = output['label']     # (32, 512, 512, 1)
                output_np = output['output']   # (32, 512, 512, 21)

                # 2. 모든 이미지 다 저장
                batch_size = output_np.shape[0]

                iou_batch = []
                acc_batch = []

                for j in range(batch_size):
                    # (512, 512, 3)
                    img_input = (input_np[j] * 255).astype(np.uint8)
                    # (512, 512, 1)
                    img_label = label_np[j]
                    # (512, 512, 21)
                    img_output = output_np[j]

                    global_idx = (i-1)*self.batch_size + j
                    input_path = os.path.join(
                        save_input_dir, f"test_{global_idx}.npy")
                    output_path = os.path.join(
                        save_output_dir, f"test_{global_idx}.npy")
                    target_path = os.path.join(
                        save_target_dir, f"test_{global_idx}.npy")
                    
                    # 3. 그림 저장
                    np.save(input_path, img_input)      # (512, 512, 3)
                    np.save(output_path, img_output)    # (512, 512, 21)
                    np.save(target_path, img_label)     # (512, 512, 1)

                    global_idx += 1

                    img_output = np.argmax(img_output, axis=-1)
                    img_output = np.expand_dims(img_output, axis=-1)

            print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_test)))
           

    def get_iou(self, label, output):

        SMOOTH = 1e-6

        # (512, 512) 로 차원 축소
        label = label.squeeze(-1)
        output = output.squeeze(-1)

        # ignore index 처리
        mask = (label != 255)
        label = label[mask]
        output = output[mask]

        # 클래스 리스트 추출 (배경 클래스 0 제외)
        lst_label_class = np.unique(label)
        lst_class = lst_label_class[lst_label_class != 0]

        iou_list = []  # 클래스별 IoU 저장

        # 클래스별 Intersection, Union 계산
        for class_ in lst_class:
            intersection = ((output == class_) & (label == class_)).sum()
            union = ((output == class_) | (label == class_)).sum()

            iou = (intersection + SMOOTH) / (union + SMOOTH)  # IoU 계산
            iou_list.append(iou)

        # 클래스별 IoU 평균 계산
        if len(iou_list) == 0:
            return 0.0  # 예측된 클래스가 없는 경우 0 반환
        else:
            return np.mean(iou_list)
