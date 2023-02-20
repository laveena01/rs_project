import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from expr.train import device
import numpy as np
import random

import sys

sys.path.append('../')

__all__ = [
    'model',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=20, input_size=448):
        super(VGG, self).__init__()

        self.num_maps = 8
        self.cos_alpha = 0.1

        self.features = features[0:]

        self.fc4_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
        )
        self.fc4_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
        )
        self.cls4 = nn.Conv2d(1024, num_classes * self.num_maps, kernel_size=1, padding=0)

        self.fc5_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
        )
        self.fc5_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
        )
        self.cls5 = nn.Conv2d(1024, num_classes * self.num_maps, kernel_size=1, padding=0)  #
        self._initialize_weights()

        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.input_size = input_size

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x, label=None):
        x = self.features[0:24](x)
        # batch_size = x.size(0)
        # size = int(self.input_size / 16)
        # parentResult = self.fc4_1(x)
        # parentResult = self.fc4_2(parentResult)
        # parentResult_multimaps = self.cls4(parentResult).view(batch_size, self.num_classes, self.num_maps, size, size)
        # parentResult = torch.sum(parentResult_multimaps, 2).view(batch_size, self.num_classes, size, size)/self.num_maps
        self.parent_map = x
        # peak_list, aggregation_parent = None, F.adaptive_avg_pool2d(parentResult, 1).squeeze(2).squeeze(2)
        x = self.features[24:](x)
        batch_size = x.size(0)
        size = int(self.input_size / 32)
        childResult = self.fc5_1(x)
        childResult = self.fc5_2(childResult)

        childResult_multimaps = self.cls5(childResult).view(batch_size, self.num_classes, self.num_maps, size, size)
        childResult = torch.sum(childResult_multimaps, 2).view(batch_size, self.num_classes, size, size) / self.num_maps

        self.salience_maps = childResult_multimaps
        self.child_map = childResult_multimaps

        peak_list, aggregation_child = None, F.adaptive_avg_pool2d(childResult, 1).squeeze(2).squeeze(2)

        return aggregation_child

    def calculate_cosineloss(self, maps):
        batch_size = maps.size(0)
        num_maps = maps.size(1)
        channel_num = int(self.num_maps/2)
        eps = 1e-8
        random_seed = random.sample(range(num_maps), channel_num)
        maps = maps[:, random_seed, :, :].view(batch_size, channel_num, -1)

        # maps_max = maps.max(dim=2)[0].expand(maps.shape[-1], batch_size, channel_num).permute(1, 2, 0)
        # maps = maps/maps_max
        X1 = maps.unsqueeze(1)
        X2 = maps.unsqueeze(2)
        dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
        # print(dot12)
        dist = dot12 / (torch.sqrt(dot11 * dot22 + eps))
        tri_tensor = ((torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1]*channel_num))).expand(batch_size, channel_num, channel_num))
        tri_tensor = tri_tensor.to(device)
        dist = dist.to(device)
        dist_num = abs((tri_tensor*dist).sum(1).sum(1)).sum()/(batch_size*channel_num*(channel_num-1)/2)

        return dist_num, random_seed


    def get_loss(self, logits, gt_labels):
        child_logits = logits
        gt_label_random = torch.zeros(gt_labels.size(0))
        for i in range(gt_labels.size(0)):
            gt_label_random[i] = random.sample(list(torch.nonzero(gt_labels[i])[:, 0]), 1)[0]

        batch_size = self.child_map.size(0)
        size = int(self.input_size / 32)

        label_index = []
        for i in range(gt_labels.size(0)):
            label_index.append(list(torch.nonzero(gt_labels[i])))
        map_temp = self.child_map.reshape(batch_size * self.num_classes, self.num_maps, size, size)
        temp = torch.zeros((batch_size, self.num_maps, size, size)).to(device)

        for i in range(batch_size):
            for k in label_index[i]:
                temp[i] += map_temp[i * self.num_classes + k].squeeze(0)

        # maps = torch.cat((
        #     (self.child_map.reshape(batch_size * self.num_classes, self.num_maps, size, size)[
        #      [gt_label_random[i].long() + (i * self.num_classes) for i in range(batch_size)], :, :, :]).reshape(batch_size,
        #                                                         self.num_maps, size, size),), 1)

        maps = torch.cat((temp,), 1)

        loss_cos, random_seed = self.calculate_cosineloss(maps)

        loss_cls_child = F.multilabel_soft_margin_loss(child_logits, gt_labels)

        loss_val = loss_cls_child * 0.7 + self.cos_alpha * loss_cos

        return loss_val #, loss_cls, loss_cos


    def get_salience_maps(self):
        return torch.mean(F.relu(self.parent_map), dim=2), torch.mean(F.relu(self.salience_maps), dim=2)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'O': [64, 64, 'L', 128, 128, 'L', 256, 256, 256, 'L', 512, 512, 512, 'L', 512, 512, 512, 'L']
}


def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    # print(model)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        # print(pretrained_dict)
        print('load pretrained model from {}'.format(model_urls['vgg16']))
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from vgg16'.format(k))
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for DA Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    # pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(pretrained=True, num_classes=12)
    model = model.to(device)
    x = torch.randn(4, 3, 448, 448)
    x = x.to(device)
    output = model(x)
    # print(output[0].shape, output[1].shape)
    loss = model.get_loss(output, torch.randn(output.shape))
    # cam, _ = model.get_salience_maps()
    # cam = cam.squeeze(0)
    # cam = cam.sum(0).cpu().detach().numpy()
    print(loss)
    # print(model)
