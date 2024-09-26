import torch
import torch.nn.functional as F

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=20.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().cuda()
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * mae.mean()