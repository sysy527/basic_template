import os
import torch
import glob
import re
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader



class CustomedDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transform = None):
        dir_data = cfg["dir_data"]
        name_data = cfg["name_data"]
        mode = cfg["mode"]

        mode = mode
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
    num_val = int(0.2 * len(full_dataset))

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42))

    loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return loader_train, loader_val
        
        