import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import collections
from torchvision import transforms, utils
from PIL import Image
from collections import OrderedDict
import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
_categories = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'Dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

categories_index = {
    'airplane': 0, 'airport': 1, 'baseballfield': 2, 'basketballcourt': 3, 'bridge': 4,
    'chimney': 5, 'dam': 6, 'Expressway-Service-area': 7, 'Expressway-toll-station': 8,
    'golffield': 9, 'groundtrackfield': 10, 'harbor': 11, 'overpass': 12, 'ship': 13,
    'stadium': 14, 'storagetank': 15, 'tenniscourt': 16, 'trainstation': 17,
    'vehicle': 18, 'windmill': 19
}
class DIORDataset(Dataset):
    def __init__(self,
                 dior_root='/data/xxxujian/PycharmProjects/dataset/DIOR',
                 image_set="train",
                 transform=None):
        self.transform = transform
        self.categories = _categories
        if image_set == "train" or image_set == "val":
            image_dir = os.path.join(dior_root, 'JPEGImages-trainval')
        else:
            image_dir = os.path.join(dior_root, 'JPEGImages-test')
        splits_dir = os.path.join(dior_root, 'ImageSets/Main')
        annotation_dir = os.path.join(dior_root, 'Annotations')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        self.image_labels = self._read_annotations()
        assert (len(self.images) == len(self.annotations))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, target = self.image_labels[index]
        img = Image.open(img_path).convert('RGB')
        # target = self.parse_voc_xml(
        #     ET.parse(self.annotations[index]).getroot())


        if self.transform is not None:
            img = self.transform(img)

        return img_path, img, target


    def _read_annotations(self):
        class_labels = OrderedDict()
        for i in range(len(self.images)):
            node = ET.parse(self.annotations[i]).getroot()
            target = self.parse_voc_xml(node)
            objects = target['annotation']['object']
            cnt = []
            if not isinstance(objects, list):
                cnt.append(objects['name'])
            else:
                for object in objects:
                    name = object['name']
                    if name not in cnt:
                        cnt.append(name)
            for name in cnt:
                img = self.images[i]
                if img not in class_labels:
                    class_labels[img] = np.zeros(len(_categories))
                class_labels[img][categories_index[name]] = 1

        return list(class_labels.items())

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def get_categories(self):
        return _categories

def show_object_rect(image, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv2.rectangle(image_show, pt1, pt2, (0, 0, 255), 2)

def show_object_name(image, name, p_tl):
    return cv2.putText(image, name, p_tl, 1, 1, (255, 255, 0))

if __name__=="__main__":
    classes_cnt = {}
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    input_size = (int(512), int(512))
    crop_size = (int(448), int(448))

    # transformation for training set
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 256
                                     transforms.RandomCrop(crop_size),  # 224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])
    dior_trainset = DIORDataset(image_set='train', transform=tsfm_train)
    dataloaders = DataLoader(dior_trainset, batch_size=4, shuffle=True, num_workers=1)
    for batch_idx, (_, inputs, targets) in enumerate(dataloaders):
        print(inputs.shape, targets.shape)

    # for i, sample in enumerate(dior_trainset, 1):
    #     image, annotation = sample[0], sample[1]['annotation']
    #     objects = annotation['object']
    #     show_image = np.array(image)
    #     # print('{} object:{}'.format(i, len(objects)))
    #     if not isinstance(objects, list):
    #         object_name = objects['name']
    #         if object_name not in classes_cnt.keys():
    #             classes_cnt[object_name] = 0
    #         classes_cnt[object_name] = classes_cnt[object_name] + 1
    #         object_bndbox = objects['bndbox']
    #         x_min = int(object_bndbox['xmin'])
    #         y_min = int(object_bndbox['ymin'])
    #         x_max = int(object_bndbox['xmax'])
    #         y_max = int(object_bndbox['ymax'])
    #         show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
    #         show_image = show_object_name(show_image, object_name, (x_min, y_min))
    #     else:
    #         for j in objects:
    #             object_name = j['name']
    #             if object_name not in classes_cnt.keys():
    #                 classes_cnt[object_name] = 0
    #             classes_cnt[object_name] = classes_cnt[object_name] + 1
    #             object_bndbox = j['bndbox']
    #             x_min = int(object_bndbox['xmin'])
    #             y_min = int(object_bndbox['ymin'])
    #             x_max = int(object_bndbox['xmax'])
    #             y_max = int(object_bndbox['ymax'])
    #             show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
    #             show_image = show_object_name(show_image, object_name, (x_min, y_min))
    #
    #     cv2.imshow('image', show_image)
    #     cv2.waitKey(0)
    # print(len(classes_cnt))
    print(len(dataloaders))
