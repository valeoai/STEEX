import torchvision.models as models
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class DenseNet121(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.feat_extract = models.densenet121(pretrained=pretrained)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024

    def forward(self, x):
        return self.feat_extract(x)


class DecisionDensenetModel(nn.Module):

    def __init__(self, num_classes=40, pretrained=False):
        super().__init__()
        self.feat_extract = DenseNet121(pretrained=pretrained)
        self.classifier = nn.Linear(self.feat_extract.output_size, num_classes)

    def forward(self, input, before_sigmoid=False):

        feat = self.feat_extract(input)
        scores = self.classifier(feat)
        proba = torch.sigmoid(scores)
        if before_sigmoid:
            return scores
        return proba
