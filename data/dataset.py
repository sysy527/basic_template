import os
import torch
import glob
import re
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms 



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
        
        image = Image.open(image_path)
        image_array = np.array(image)
        
        if image_array.ndim == 2:
            image_array = image_array[:, :, np.newaxis]    
        
        if self.transform:
            image_array = self.transform(image_array)
        
        label = torch.tensor(label, dtype=torch.long)
        data = {'image': image_array, 'label': label}
            
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
        dir_data = cfg["dir_data"]    # '/home/seyeon/datasets
        name_data = cfg["name_data"]  #  pascalVOC2012
        mode = cfg["mode"]            #  train
        
        path_data_dir = os.path.join(dir_data, name_data, mode)
        self.transform = transform
        
        dir_label = os.path.join(path_data_dir, 'label_onehot')
        dir_image = os.path.join(path_data_dir, 'input')
        #dir_palette = "/home/seyeon/datasets/VOCdevkit/VOC2012/SegmentationClass/" 
        
        lst_filename = os.listdir(dir_image)
        
        self.data = []
        for filename in lst_filename:
            img_path = os.path.join(dir_image, filename)
            label_path = os.path.join(dir_label, filename.replace(".jpg", ".npy")) # ver1 npy, 나머지 png
            self.data.append((img_path, label_path)) 
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        image_path = self.data[idx][0]
        label_path = self.data[idx][1]
        
        image = Image.open(image_path)
        image = image.resize((512,512), Image.NEAREST)
        #label = Image.open(label_path)
        #label = label.resize((512,512), Image.NEAREST)
    
        image_arr = np.array(image)
        #label_arr = np.array(label)
        label_arr = np.load(label_path)
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
    

class Normalization(object):
    def __init__(self, mean = 0.5, std = 0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        label, input = data['label'], data['image']
        input = input / 255.0
        input = (input - self.mean) / self.std
        
        data = {'label': label, 'image': input}
        
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
    
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['image']
        
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
            
        data = {'label': torch.from_numpy(label), 'image': torch.from_numpy(input)}
        
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