import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
class FC_vgg(nn.Module):
    def __init__(self, model, num_classes):
        super(FC_vgg, self).__init__()
        self.features = model.features
        num_features = 512
        self.cls = self.classifier(num_features, num_classes)

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

    def forward(self, x):
        x = self.features[:24](x)

        self.parent_map = x
        x = self.features[24:](x)

        out = self.cls(x)
        self.salience_maps = out
        peak_list, aggregation = None, F.adaptive_avg_pool2d(out, 1).squeeze(2).squeeze(2)
        return aggregation

    def get_loss(self, logits, gt_labels):
        # loss_cls = self.loss_cross_entropy(logits0, gt_labels.long())

        logits0 = logits
        loss_cls = F.multilabel_soft_margin_loss(logits0, gt_labels)
        loss_val = loss_cls
        return loss_val


    def get_salience_maps(self):
        return self.parent_map, self.salience_maps

def model(pretrained=True, num_classes=10):
    model = models.vgg16(pretrained=pretrained)
    model_ft = FC_vgg(model, num_classes=num_classes)
    return model_ft

if __name__ == '__main__':
    model_ft = model(True, 10)
    print(model_ft)