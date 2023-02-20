import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import torch
from expr.train import device

class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_ResNet, self).__init__()
        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # # classifier
        # self.cls = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
        # )
        # self.cls_erase = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
        # )
        # num_features = model.layer4[1].conv1.in_channels
        self.cls = self.classifier(512, num_classes)

        self.cls_erase = self.classifier(512, num_classes)


        # threshold
        self.threshold = 0.6
        # loss
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  # fc8
        )

    def forward(self, x, label=None, epoch=-2):
        self.img_erased = x
        x = self.features[0:7](x)
        self.parent_map = x
        x = self.features[7](x)

        feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # out
        out = self.cls(x)
        self.salience_maps = out
        logits_1 = torch.mean(torch.mean(out, dim=2), dim=2)

        # # erase
        if epoch > -1:
            localozation_map_normed = self.get_atten_map(out, label, True)
            self.attention = localozation_map_normed
            feat_erase = self.erase_feature_maps(localozation_map_normed, feat, self.threshold)
            out_erase = self.cls_erase(feat_erase)
            logits_ers = torch.mean(torch.mean(out_erase, dim=2), dim=2)

            return [logits_1, logits_ers]                                                                                                                                                                                                                                                                                                    d


        return [logits_1]

    def get_loss(self, logits, gt_labels):

        loss_cls = F.multilabel_soft_margin_loss(logits[0], gt_labels.long())
        loss_cls_ers = F.multilabel_soft_margin_loss(logits[1], gt_labels.long())
        loss_val = loss_cls + loss_cls_ers
        return loss_val

    def get_salience_maps(self):
        return self.parent_map, self.salience_maps


    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        # airplane
        label = gt_labels.long()

        # dior
        label_index = []
        for i in range(gt_labels.size(0)):
            label_index.append(list(torch.nonzero(gt_labels[i])))

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.to(device))


        for batch_idx in range(batch_size):
            # DIOR
            temp = torch.zeros((1, 14, 14)).to(device)
            for k in label_index[batch_idx]:
                temp = temp + feature_maps[batch_idx, k, :, :]

            # airplane
            # temp = feature_maps[batch_idx, label.data[batch_idx], :, :]

            atten_map[batch_idx, :, :] = torch.squeeze(temp)

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map

    def erase_feature_maps(self, atten_map_normed, feature_maps, threshold):
        # atten_map_normed = torch.unsqueeze(atten_map_normed, dim=1)
        # atten_map_normed = self.up_resize(atten_map_normed)
        if len(atten_map_normed.size()) > 3:
            atten_map_normed = torch.squeeze(atten_map_normed)
        atten_shape = atten_map_normed.size()

        pos = torch.ge(atten_map_normed, threshold)
        mask = torch.ones(atten_shape).to(device)
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)
        # erase
        erased_feature_maps = feature_maps * Variable(mask)

        return erased_feature_maps

def model(pretrained=True, num_classes=10):
    model = models.resnet34(pretrained=pretrained)
    model_ft = FC_ResNet(model, num_classes=num_classes)
    return model_ft


# import torch.optim as optim
if __name__ == '__main__':
    model_ft = model(True, 10)
    model_ft = model_ft.to(device)
    x = torch.randn(1, 3, 448, 448).to(device)
    output = model_ft(x, torch.randn(1).to(device))

    loss = model_ft.get_loss(output, torch.randn(1).to(device), 60)
    _, cam = model_ft.get_salience_maps()
    print(loss, _.shape, cam.shape)
