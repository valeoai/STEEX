import torchvision.models as models
import torch
import torch.nn as nn


class DecisionResnetModel(nn.Module):

  def __init__(self,num_classes, resnet=50, pretrained=True, layers_to_freeze=[]):
    super(DecisionResnetModel,self).__init__()
    if resnet==18:
      self.resnet = models.resnet18(pretrained=pretrained)
    elif resnet==34:
      self.resnet = models.resnet34(pretrained=pretrained)
    elif resnet==50:
      self.resnet = models.resnet50(pretrained=pretrained)
    elif resnet==101:
      self.resnet = models.resnet101(pretrained=pretrained)
    elif resnet==152:
      self.resnet = models.resnet152(pretrained=pretrained)

    for layer in layers_to_freeze:

      if layer == 'conv1':
        for param in self.resnet.conv1.parameters():
          param.requires_grad = False
      if layer == 'bn1':
        for param in self.resnet.bn1.parameters():
          param.requires_grad = False
      if layer == 'layer1':
        for param in self.resnet.layer1.parameters():
          param.requires_grad = False
      if layer == 'layer2':
        for param in self.resnet.layer2.parameters():
          param.requires_grad = False
      if layer == 'layer3':
        for param in self.resnet.layer3.parameters():
          param.requires_grad = False
      if layer == 'layer4':
        for param in self.resnet.layer4.parameters():
          param.requires_grad = False

    num_features = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(num_features, num_classes)

  def forward(self, input, before_sigmoid=False):

    scores = self.resnet(input)
    proba = torch.sigmoid(scores)
    if before_sigmoid:
        return scores
    return proba
