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

        for key, value in self.cfg.items():
            setattr(self, key, value)

        gpu_id = 3
        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
        self.dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name, 'ver1')
        self.dir_log = os.path.join(self.dir_log, self.scope, self.name_data, self.model_name, 'ver1')
        self.data_filepath = os.path.join(self.dir_data, self.name_data, self.mode)

        self.fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        self.fn_denorm = lambda x, mean, std: (x * std) + mean
        # isbi
        '''
        self.fn_class = lambda x: 1.0 * (x > 0.5)     
        '''                        # BCEwithlogitLoss
        # pascal
        self.fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True).float()  # softmax(2) -> cross entropyloss
 

    def train_one_epoch(self, epoch, loader_train, net, optim, loss_fn, writer_train):
        num_batch_train = len(loader_train)

        net.train()
        loss_train = []

        for i, batch in enumerate(loader_train, 1):
            def should(freq):
                return freq > 0 and (i % freq == 0 or i == num_batch_train)
            # pascal
           
            input = batch['image'].to(self.device).type(torch.float32)
            label = batch['label'].to(self.device).type(torch.float32)  #float32 ver3/ver2일땐 long
          
            # isbi
            '''
            input = batch['image'].to(self.device)
            label = batch['label'].to(self.device)
            
            output = net(input)
            '''


            #### randomcrop 시, "0" 이 50프로 넘는거만 학습에 넣기 

            ### ver2. index 그대로 / output(1) channel / Cross entropy loss
            '''
            output = net(input)
            '''
            ### ver3. one-hot encoding / output(21) channel , softmax 적용 X / output에 log softmax 적용 / nll loss
            
            output = net(input)
            output = F.log_softmax(output, dim=1)  # NLL loss - log softmax이 입력값이어야
            
            # backward netD
            optim.zero_grad()
            ### ver0. 
            '''
            loss = loss_fn(output, label)  
            '''
            ### ver2. 
            '''
            loss = loss_fn(output, label.squeeze(1))  
            '''
    
            ### ver1,ver3. softmax 적용 X 
           
            loss = loss_fn(output, label.argmax(dim=1)) 
          

            # get losses
            loss.backward()
            optim.step()
            loss_train += [loss.item()]

            print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f' % (epoch, i, num_batch_train, np.mean(loss_train)))

            ### isbi
            '''
            label = self.fn_tonumpy(label)
            input = self.fn_tonumpy(self.fn_denorm(input, mean=0.5, std=0.5))
            output = self.fn_tonumpy(self.fn_class(output))
            '''

            ### ver2.
            '''
            label_np = self.fn_tonumpy(self.fn_class(label))
            input_np = self.fn_tonumpy(self.fn_denorm(input, mean=0.5, std=0.5))
            output_np = self.fn_tonumpy(output)
            '''

            ### ver3. 
            '''
            label_np = self.fn_tonumpy(label)
            input_np = self.fn_tonumpy(self.fn_denorm(input, mean=0.5, std=0.5))
            output_np = self.fn_tonumpy(output)
            '''
            
            # writer_train.add_image('label', label_np, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            # writer_train.add_image('input', input_np, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
            # writer_train.add_image('output', output_np, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

        return np.mean(loss_train)

    def val_one_batch(self, batch, net, loss_fn):

        # Set model to evaluation mode
        net.eval()

        # pascal
   
        input = batch['image'].to(self.device).type(torch.float32)
        label = batch['label'].to(self.device).type(torch.long)   # torch.Size([32, 1, 512, 512])
     

        # isbi
        '''
        input = batch['image'].to(self.device)
        label = batch['label'].to(self.device)
        '''

        with torch.no_grad():
            # ver2.
            '''
            output = net(input)
            '''                        # torch.Size([32, 21, 512, 512])

            # ver3.
         
            output = net(input)
            output = F.log_softmax(output, dim=1)
       

            # Calculate loss
            # isbi
            '''
            loss = loss_fn(output, label)
            '''
            # ver2. index 기반, cross-entropy loss
            '''
            loss = loss_fn(output, label.squeeze(1))
            '''
            # ver3. one-hot encoding / output(21) channel , softmax 적용 X / output에 log softmax 적용 / nll loss
    
            loss = loss_fn(output, label.argmax(dim=1))
   

        return loss.item(), output

    def train(self, loader_train, loader_val, net, optim, loss_fn):
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name, 'ver1')
        # setup tensorboard
        writer_train = SummaryWriter(log_dir=os.path.join(self.dir_log, 'train'))
        writer_val = SummaryWriter(log_dir=os.path.join(self.dir_log, 'val'))

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

                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f' % (epoch, i, num_batch_val, np.mean(loss_val)))

                    ### ver0. isbi
                    '''
                    label = self.fn_tonumpy(batch['label'])
                    input = self.fn_tonumpy(self.fn_denorm(batch['image'], mean=0.5, std=0.5))
                    output = self.fn_tonumpy(self.fn_class(output))
                    '''
                    ##### ver1. Customed One-hot 사용, npy 파일 이용해서 진행 #####

                    ##### ver2. index 기반 Cross-Entropy loss 사용 #####
                    '''
                    label_np = self.fn_tonumpy((batch['label']))
                    input_np = self.fn_tonumpy(self.fn_denorm(batch['image'], mean=0.5, std=0.5))
                    output_np = self.fn_tonumpy(output)
                    '''
                    ##### ver3. 내장함수 onehot 사용, NLL loss 사용 #####
                    '''
                    label_np = self.fn_tonumpy(batch['label'])
                    input_np = self.fn_tonumpy(self.fn_denorm(batch['image'], mean=0.5, std=0.5))
                    output_np = self.fn_tonumpy(output)
                    '''                  

                    #writer_val.add_image('label', label_np, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                    #writer_val.add_image('input', input_np, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                    #writer_val.add_image('output', output_np, num_batch_val * (epoch - 1) + i, dataformats='NHWC')

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
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, self.name_data, self.model_name, 'ver1')
        num_batch_test = len(loader_test)

        # load from checkpoints
        net, loaded_epoch = self.load(dir_chck, net, mode=self.mode)
        print('Epoch :', loaded_epoch)

        save_input_dir = "./result/ver1/input"
        save_output_dir = "./result/ver1/output"
        save_target_dir = "./result/ver1/target"

        os.makedirs(save_input_dir, exist_ok=True)
        os.makedirs(save_output_dir, exist_ok=True)
        os.makedirs(save_target_dir, exist_ok=True)  # 저장 폴더 생성

        # test phase
        with torch.no_grad():
            net.eval()

            loss_test = []
            acc_test = []
            iou_test = []

            for i, batch in enumerate(loader_test, 1):
                # isbi
                '''
                input = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                '''
                # pascal
             
                input = batch['image'].to(self.device).type(torch.float32)  # 입력 이미지 (batch_size, 3, 512, 512)
                label = batch['label'].to(self.device).type(torch.long)  # 정답 (batch_size, 21, 512, 512)
           
                loss, output = self.val_one_batch(batch, net, loss_fn)

                # sliding window patch inference

                loss_test.append(loss)

                print('TEST: BATCH %04d/%04d: LOSS: %.4f' % (i, num_batch_test, np.mean(loss_test)))

                ### ver 2. 
                '''
                input_np = self.fn_tonumpy(self.fn_denorm(input, mean=0.5, std=0.5))  # (batch, 512, 512, 3)
                label_np = self.fn_tonumpy(label)                                     # (32, 512, 512, 1)                        
                output_np = self.fn_tonumpy(output)                                   # (32, 512, 512, 21)
                '''
                ### ver 3.
               
                input_np = self.fn_tonumpy(self.fn_denorm(input, mean=0.5, std=0.5))  # (batch, 512, 512, 3)
                label_np = self.fn_tonumpy(self.fn_class(label))                      # (32, 512, 512, 1)
                output_np = self.fn_tonumpy(torch.exp(output))                        # (32, 512, 512, 21) -> softmax 까지된 상태
             

                # 2. 배치에서 마지막 이미지만 선택
                batch_size = output_np.shape[0]
                iou_batch = []
                #acc_batch = []
                for j in range(batch_size):

                    ### ver2
                    '''   
                    img_input = (input_np[j] * 255).astype(np.uint8)   # (512, 512, 3)
                    img_label = label_np[j]                            # (512, 512, 1)
                    img_output = output_np[j]                          # (512, 512, 21)
                    '''

                    ### ver3
            
                    img_input = (input_np[j] * 255).astype(np.uint8)   # (512, 512, 3)
                    img_label = label_np[j]                            # (512, 512, 1)
                    img_output = output_np[j]                           #(512, 512, 21)
                   

                    input_path = os.path.join(save_input_dir, f"test_{((i-1)*batch_size)+j}.npy")
                    output_path = os.path.join(save_output_dir, f"test_{((i-1)*batch_size)+j}.npy")
                    target_path = os.path.join(save_target_dir, f"test_{((i-1)*batch_size)+j}.npy")

                    # 3. 그림 저장
                    np.save(input_path, img_input)      # (512, 512, 3)
                    np.save(output_path, img_output)    # (512, 512, 21)
                    np.save(target_path, img_label)     #(512, 512, 1)

                    total = img_label.shape[0] * img_label.shape[1] * img_label.shape[2]
                    #correct = (img_label == img_output).sum().item()
                    #accuracy = (correct / total)
                    #acc_test.append(accuracy)
                    #acc_batch.append(accuracy)

                    img_output = np.argmax(img_output, axis=-1)
                    img_output = np.expand_dims(img_output, axis=-1)
                    iou = self.get_iou(img_label, img_output)
                    iou_batch.append(iou)
                    iou_test.append(iou)

                #print('TEST: BATCH %04d/%04d: ACC: %.4f' % (i, num_batch_test, np.mean(acc_batch)))
                print('TEST: BATCH %04d/%04d: IOU: %.4f' % (i, num_batch_test, np.mean(iou_batch)))

            print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_test)))
            #print('TEST: AVERAGE ACC: %.6f' % (np.mean(acc_test)))
            print('TEST: AVERAGE IOU: %.6f' % (np.mean(iou_test)))

    def get_iou(self, label, output):

        SMOOTH = 1e-6

        label = label.squeeze(-1)
        output = output.squeeze(-1)

        lst_label_class = np.unique(label)
        del_class = np.argwhere(lst_label_class == 0)
        lst_class = np.delete(lst_label_class, del_class)

        intersections = 0
        for class_ in lst_class:
            intersection = ((output.astype(int) == class_) & (label.astype(int) == class_)).sum((0, 1))
            intersections += intersection

        union = (output.astype(int) | label.astype(int)).sum((0, 1))
        iou = (intersections + SMOOTH) / (union + SMOOTH)

        return iou
