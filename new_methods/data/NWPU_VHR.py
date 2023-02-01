from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from PIL import Image
import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F


_categories = ['airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
           'basketball_court', 'ground_track_field', 'harbor', 'bridge']
categories_index = {'airplane': 1, 'ship': 2, 'storage_tank': 3, 'baseball_diamond': 4, 'tennis_court': 5,
           'basketball_court': 6, 'ground_track_field': 7, 'harbor': 8, 'bridge': 9}

class VHR_10(Dataset):
    def __init__(self, root='/data/xxxujian/PycharmProjects/dataset/NWPU VHR-10 dataset_split_1',
                 image_set='train',  transform=None):

        self.root = root
        self.dir_mark = os.path.join(self.root, 'ground truth')
        self.dir_image = os.path.join(self.root, image_set)
        self.categories = _categories
        self.image_labels = self.readtxt()

        self.transform = transform


    def __len__(self):
        return len(self.image_labels)
    def __getitem__(self, index):
        '''
        image
        :param index:
        :return:
        '''
        filename, target = self.image_labels[index]
        target = torch.from_numpy(target).float()
        img = Image.open(os.path.join(
            self.dir_image, filename + '.jpg')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return os.path.join(self.dir_image, filename + '.jpg'), img, target

    def readtxt(self):
        class_labels = OrderedDict()
        for path in os.listdir(self.dir_image):
            img = path.split('.')[0]
            if int(img) > 650:
                if img not in class_labels:
                    class_labels[img] = np.zeros(len(_categories))
                class_labels[img][0] = 1
            else:
                with open(os.path.join(self.dir_mark, img+'.txt'), 'r') as f:
                    contents = f.read()
                    names = []
                    cnt = []
                    objects = contents.split('\n')
                    for i in range(objects.count('')):
                        objects.remove('')
                    for objecto in objects:
                        name = objecto.split(',')[4]
                        name = name.strip()
                        if int(name) == 10:
                            break
                        name = _categories[int(name) - 1]
                        names.append(name)
                        if name not in cnt:
                            cnt.append(name)
                for name in cnt:
                    if img not in class_labels:
                        class_labels[img] = np.zeros(len(_categories))
                    class_labels[img][categories_index[name] - 1] = 1
        return list(class_labels.items())

if __name__ == "__main__":
    vhr10_Dataset = VHR_10()
    # for i, sample in enumerate(vhr10_Dataset, 1):
    #     img, label = sample
    #     print(label)
    print(len(vhr10_Dataset))
        # print(img, end=' ')
        # print(label)