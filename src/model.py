import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model