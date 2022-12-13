from torchvision.datasets import VisionDataset
import torch

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class PACS(VisionDataset):
    def __init__(self, domain, root, transform=None, target_transform=None):
        super(PACS, self).__init__(domain, transform=transform, target_transform=target_transform)
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        self.domain = domain
        self.root = root
        # labels = [name for name in os.listdir(f"{root}/PACS/{domain}") if os.path.isdir(f"{root}/{domain}/{name}")]
        self.labels_dict = {}
        self.data = []
        self.labels = []
        
        with open(f'{root}/txt_lists/{self.domain}.txt', 'r') as f: 
          img_set = f.readlines()
          #img_set = list(filter(lambda img_path: (img_path.split("/")[0] != "BACKGROUND_Google"), img_set))
          self.data = [pil_loader(f"{root}/PACS/{img_path.split(' ')[0]}") for img_path in img_set]
          self.labels = [int(img_path.split(" ")[1]) for img_path in img_set]
        
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data[index], self.labels[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.labels) # Provide a way to get the length (number of elements) of the dataset
        return length
    
    def stratified_sample(self, n_groups):
      import sklearn.model_selection
      return sklearn.model_selection.train_test_split([i for i in range(len(self.labels))], train_size = 0.5, test_size = 0.5, stratify = self.labels)
      '''sample_idxs = [[] for i in range(n_groups)]
      for key, class_indices in self.img_idx_per_class.items():
        size = len(class_indices) // n_groups
        random_idx = torch.randperm(len(class_indices))
        start = 0
        end = size
        print(key)
        for i in range(n_groups):
          sampling = torch.index_select(torch.LongTensor(class_indices), 0, torch.narrow(random_idx,0,start,size) )
          sample_idxs[i] += sampling.tolist()
          start += size
          print(sampling.tolist())
      print(sample_idxs)
      return sample_idxs
      '''
