import os
import torch
import glob
import re
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms 
import matplotlib.pyplot as plt
import cv2

class CustomedDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transform = None):
        dir_data = cfg["dir_data"]
        name_data = cfg["name_data"]
        mode = cfg["mode"]
        
        self.transform = transform
        
        path_data_dir = os.path.join(dir_data, name_data, mode)
        lst_dataset_label = os.listdir(path_data_dir) # [0, 1, ... ,9]

        self.data = []
        for file_label in lst_dataset_label:
            path_labeled_file_lst = path_data_dir + '/' + file_label # /home/seyeon/datasets/mnist/train/0
            lst_labeled_file = os.listdir(path_labeled_file_lst)
            
            for filename in lst_labeled_file:
                full_path = os.path.join(path_labeled_file_lst, filename)
                self.data.append([file_label,full_path])  # dict append 
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image_path = self.data[idx][1]
        label = int(self.data[idx][0])
        image = plt.imread(image_path)
        
        if image.ndim == 2:
            image = image[:, :, np.newaxis]    
        
        image = Image.fromarray((image.squeeze() * 255).astype(np.uint8))  
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        data = {'label': label, 'image': image}
            
        return data
            
# dataset을 받아서 데이터 로더로 반환 # utils 에 넣어도 되지만
def buildDataLoader(full_dataset, cfg):
    cfg = cfg["train"]
    batch_size = cfg["batch_size"]
    
    num_train = int(0.8 * len(full_dataset))
    num_val = len(full_dataset) - num_train

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42))

    loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return loader_train, loader_val
        
        
class CustomedDataset_Seg(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transform = None):
        dir_data = cfg["dir_data"]    # '/home/seyeon/datasets
        name_data = cfg["name_data"]  #  VOCdevkit/VOC2012/Segmentation
        mode = cfg["mode"]            #  train
        
        path_data_dir = os.path.join(dir_data, name_data, mode)
        self.transform = transform
        
        dir_label = os.path.join(path_data_dir, 'label')
        dir_image = os.path.join(path_data_dir, 'input')
        
        lst_filename = os.listdir(dir_image)
        lst_label_filename = os.listdir(dir_label)
        
        self.data = []
        for filename, related_masks in self.find_matching_files(lst_filename, lst_label_filename):
            fullpath_filename = os.path.join(dir_image, filename)
            
            for filename_mask in related_masks:
                fullpath_filename_mask = os.path.join(dir_label, filename_mask)
                self.data.append((fullpath_filename, fullpath_filename_mask))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image_path = self.data[idx][0]
        label_path = self.data[idx][1]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        #image_resized  = image.resize((256,256))    -> transform부분에
        image_arr = np.array(image)
        label_arr = np.array(label)
        
        image_arr = image_arr/255.0
        label_arr = label_arr/255.0
        
        if image_arr.ndim == 2:
            image_arr = image_arr[:, :, np.newaxis]
        
        if label_arr.ndim == 2:
            label_arr = label_arr[:, :, np.newaxis]
           
        data = {'image': image_arr, 'label': label_arr} 
            
        if self.transform:
            data = self.transform(data)
            
        return data
    
    def find_matching_files(self, lst_filename, lst_label_filename):
        """ input 이미지 파일과 대응하는 label 파일을 찾는 함수 """
        matched_files = []

        for filename in lst_filename:
            file_base = os.path.splitext(filename)[0]  # 확장자 제거 (ex: input_000, 2007_000033)

            # 🔹 Case 1: `_숫자`로 매칭 (ex: input_000.png ↔ label_000.png)
            if "_" in file_base and file_base.split("_")[-1].isdigit():
                file_num = file_base.split("_")[-1]  # `_` 이후 숫자 추출
                related_masks = [mask for mask in lst_label_filename if f"_{file_num}." in mask]

            # 🔹 Case 2: `-숫자`로 매칭 (ex: 2007_000033.png ↔ 2007_000033-1.png)
            elif "-" in file_base:
                file_base_main = file_base.split("-")[0]  # `-` 앞부분만 추출
                related_masks = [mask for mask in lst_label_filename if mask.startswith(file_base_main + "-")]

            # 🔹 기존 방식 (Case 2처럼 파일명 전체 비교)
            else:
                related_masks = [mask for mask in lst_label_filename if mask.startswith(file_base)]

            matched_files.append((filename, related_masks))  # 매칭된 파일들 저장

        return matched_files
    
class CustomedDataset_MultiClass_Seg(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transform = None):
        dir_data = cfg["dir_data"]          # '/home/seyeon/datasets
        name_data = cfg["name_data"]        #  pascalVOC2012
        self.mode = cfg["mode"]             #  train
        
        path_data_dir = os.path.join(dir_data, name_data, mode)
        self.transform = transform
        
        dir_label = os.path.join(path_data_dir, 'label')
        dir_image = os.path.join(path_data_dir, 'input')
        #dir_palette = "/home/seyeon/datasets/VOCdevkit/VOC2012/SegmentationClass/" 
        
        lst_filename = os.listdir(dir_image)
        
        self.data = []
        for filename in lst_filename:
            img_path = os.path.join(dir_image, filename)
            label_path = os.path.join(dir_label, filename.replace(".jpg", ".png")) # ver1 npy, 나머지 png
            self.data.append((img_path, label_path)) 
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image_path = self.data[idx][0]
        label_path = self.data[idx][1]
        
        image = Image.open(image_path)
        image = image.resize((256,256), Image.NEAREST)
        label = Image.open(label_path)
        label = label.resize((256,256), Image.NEAREST)
    
        image_arr = np.array(image)
        label_arr = np.array(label)

        ## ver1.
        #label_arr = np.load(label_path)
        #### ver2. crossentropy ignore index = 255
        #label_arr[label_arr==255] = 0
        #### ver3. 
        #label_arr[label_arr==255] = 0

        if image_arr.ndim == 2:
            image_arr = image_arr[:, :, np.newaxis]
        if label_arr.ndim == 2:
            label_arr = label_arr[:, :, np.newaxis]
        
        data = {'image': image_arr, 'label': label_arr} 
        
        if self.transform:
            data = self.transform(data)

        return data
'''
        if self.mode == 'train':
            # 🔹 Foreground 중심으로 Crop 🔹
            label_cropped = data['label'].squeeze(0)  # (512, 512)
            foreground_mask = (label_cropped != 0) & (label_cropped != 255)
            foreground_pixels = torch.where(foreground_mask)

            # 🔹 Foreground가 없는 경우 예외 처리 🔹
            if len(foreground_pixels[0]) == 0:
                # Foreground가 없으면 이미지 중앙에서 고정된 96×96 Crop
                crop_y_start = (512 - 96) // 2
                crop_x_start = (512 - 96) // 2
            else:
                # 🔹 Foreground 중심 계산 🔹
                y_center = (foreground_pixels[0].min() + foreground_pixels[0].max()) // 2
                x_center = (foreground_pixels[1].min() + foreground_pixels[1].max()) // 2

                # 🔹 Crop 시작 좌표 🔹
                crop_y_start = y_center - 48
                crop_x_start = x_center - 48

                # 🔹 시작 좌표가 0보다 작으면 0으로 보정 🔹
                crop_y_start = max(0, crop_y_start)
                crop_x_start = max(0, crop_x_start)

                # 🔹 시작 좌표 + 96이 이미지 크기(512)를 초과하면 시작 좌표를 조정 🔹
                if crop_y_start + 96 > 512:
                    crop_y_start = 512 - 96
                if crop_x_start + 96 > 512:
                    crop_x_start = 512 - 96

            # 🔹 Crop 끝 좌표 🔹
            crop_y_end = crop_y_start + 96
            crop_x_end = crop_x_start + 96

            # 🔹 (96, 96) 크기 보장 확인 🔹
            if (crop_y_end - crop_y_start != 96) or (crop_x_end - crop_x_start != 96):
                # 🔸 예외 처리: 중앙에서 고정된 96×96 Crop 🔸
                crop_y_start = (512 - 96) // 2
                crop_x_start = (512 - 96) // 2
                crop_y_end = crop_y_start + 96
                crop_x_end = crop_x_start + 96
                print("🔸 Warning: 중앙에서 고정된 96×96 Crop 수행")

            # 🔹 이미지 및 라벨 Crop 🔹
            data['image'] = data['image'][:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            data['label'] = data['label'][:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            # 🔹 (96, 96) 크기 확인 🔹
            #print(f"Image shape: {data['image'].shape}")
            #print(f"Label shape: {data['label'].shape}")

            # 🔹 0 크기 예외 처리 🔹
            if data['image'].shape[1] == 0 or data['image'].shape[2] == 0:
                print("🔸 Warning: 0 크기 발생, 이 데이터는 건너뜀")
            else:
                data = {'image': data['image'], 'label': data['label']} 
'''



#cycleGan - unpaired Dataset
class CustomedDataset_cycleGAN(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transform = None, data_type = 'both'):
        dir_data = cfg["dir_data"]    # '/home/seyeon/datasets
        name_data = cfg["name_data"]  # monet2photo
        self.mode = cfg["mode"]            #  train
        self.data_type = data_type
        
        path_data_dir = os.path.join(dir_data, name_data, mode)
        self.transform = transform
        self.to_tensor = ToTensor()
        
        dir_trainA = os.path.join(path_data_dir, 'A')
        dir_trainB = os.path.join(path_data_dir, 'B')
        
        lst_data_A = sorted(os.listdir(dir_trainA))
        lst_data_B = sorted(os.listdir(dir_trainB))
        self.dir_trainA = dir_trainA
        self.dir_trainB = dir_trainB
        self.lst_dataA = lst_data_A
        self.lst_dataB = lst_data_B
            
    def __len__(self):
        if self.data_type == 'both':
            if len(self.lst_dataA) < len(self.lst_dataB):
                return len(self.lst_dataA)
            else:
                return len(self.lst_dataB)
            
        elif self.data_type == 'a':
            return len(self.lst_dataA)
        elif self.data_type == 'b':
            return len(self.lst_dataB)
        
    def __getitem__(self, idx):
        
        data = {}

        if self.data_type == 'a' or self.data_type == 'both':
            data_a = plt.imread(os.path.join(self.dir_trainA, self.lst_dataA[idx]))
            
            if data_a.ndim == 2:
                data_a = data_a[:,:,np.newaxis]
            if data_a.dtype == np.uint8:
                data_a = data_a / 255.0
                
            data['data_a'] = data_a
            
        if self.data_type == 'b' or self.data_type == 'both':
            data_b = plt.imread(os.path.join(self.dir_trainB, self.lst_dataB[idx]))
            
            if data_b.ndim == 2:
                data_b = data_b[:,:,np.newaxis]
            if data_b.dtype == np.uint8:
                data_b = data_b / 255.0
                
            data['data_b'] = data_b

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # input = (input - self.mean) / self.std
        # label = (label - self.mean) / self.std
        #
        # data = {'label': label, 'input': input}

        # Updated at Apr 5 2020
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data

    
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['image']
        
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
            
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
            
        data = {'label': label, 'image': input}
        
        return data


class ToOnehot(object):
    def __call__(self, data):
        label, input = data['label'], data['image']
        
        label = label.type(torch.LongTensor)
        label = F.one_hot(label, num_classes=21)
        label = label.squeeze(dim=0)
        label= label.permute(2,0,1)
        data = {'label': label, 'image': input}
        
        return data
    
class RandomCrop(object):
    def __call__(self, data):
        label, input = data['label'], data['image']
        
        t_rand_crop = transforms.RandomCrop((96,96))

        label = t_rand_crop(label)
        input = t_rand_crop(input)
        data = {'label': label, 'image': input}
        
        return data


class ToTensor(object):
    def __call__(self, data):
        
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data
    
'''
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['image']
        
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
            
        data = {'label': torch.from_numpy(label), 'image': torch.from_numpy(input)}
        
        return data
'''   


class Resize(object):
    def __init__(self, img_load):
        self.img_load = img_load

    def __call__(self, data):
        for key, value in data.items():
            data[key] = cv2.resize(data[key], (self.img_load, self.img_load))

        return data


