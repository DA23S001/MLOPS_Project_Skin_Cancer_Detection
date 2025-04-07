import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from efficientnet_pytorch import EfficientNet

from Skin_Cancer_Classifier.entity.config_entity import PrepareBaseModelConfig


class Baseline(nn.Module):
    def __init__(self, pretrained=True, arch_name="efficientnet", num_classes=8):
        super(Baseline, self).__init__()

        self.arch_name = arch_name.lower()
        self.num_classes = num_classes

        if self.arch_name == "efficientnet":
            self.base_model = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
            nftrs = self.base_model._fc.in_features
            self.base_model._fc = nn.Linear(nftrs, self.num_classes)

        elif self.arch_name == "resnet18":
            weights = 'IMAGENET1K_V1' if pretrained else 'DEFAULT'
            self.base_model = resnet18(weights=weights)
            nftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(nftrs, self.num_classes)

        else:
            raise ValueError(f"Unsupported architecture: {arch_name}")


    def forward(self, image):
        return self.base_model(image)
    


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = Baseline(
            pretrained=self.config.pretrained,
            arch_name=self.config.params_arch_name,
            num_classes=self.config.params_classes
        )
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def save_model(path: Path, model: nn.Module):
        """
        Save the PyTorch model to the specified path
        
        Args:
            path: Path where the model should be saved
            model: PyTorch model to be saved
        """
        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model state dict
            torch.save(model.state_dict(), path)
            print(f"Model saved successfully at: {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise