import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


class FaceRecog(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.resnet50 = models.resnet50(pretrained)
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])
        # Replace last layer
        self.classifier = nn.Sequential(nn.Flatten(),
                                         nn.Linear(self.resnet50.fc.in_features, num_classes))

    def forward(self, x):
        x = self.features(x)
        y = self.classifier(x)
        return y
    
    def summary(self, input_size, device='cpu'):
        return summary(self, input_size, device=device)



if __name__ == "__main__":
    model = FaceRecog(105, True)
    model.summary((3, 224, 224))