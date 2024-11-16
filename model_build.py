import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ModelBuild:
    """
    build a custom or pretrined image classfication model
    """
    def __init__(self, model_size, num_classes,pretrained=True):
        """
        initialize the model with the model size and number of classes
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.model = None

        pass

    def _get_pretrained_model(self):
        """
        get a pretrained model
        """
        if self.model_size == '18':
            self.model = models.resnet18(pretrained=True)
        elif self.model_size == '34':
            self.model = models.resnet34(pretrained=True)
        elif self.model_size == '50':
            self.model = models.resnet50(pretrained=True)
        elif self.model_size == '101':
            self.model = models.resnet101(pretrained=True)
        elif self.model_size == '152':
            self.model = models.resnet152(pretrained=True)
        else:
            raise ValueError("Invalid model size. Choose from '18', '34', '50', '101', '152'.")

    def _add_layers(self):
        """
        add layers to the model
        """
        if self.model is None:
            self._get_pretrained_model()
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def compile(self, optimizer, loss, metrics):
        """
        compile the model with optimizer, loss and metrics
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def save(self, path):
        """
        save the model to the given path
        """
        torch.save(self.model.state_dict(), path)
