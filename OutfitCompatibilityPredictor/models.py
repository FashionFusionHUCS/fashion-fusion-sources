import torch
import torch.nn as nn
import timm
from CLIP import clip

# Define the feature extractor using a pretrained model
class ViTModel(nn.Module):
    def __init__(self, model_name='vit_small_patch32_224.augreg_in21k_ft_in1k'):
        super(ViTModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.reset_classifier(0)  # Remove the classification layer
        for param in self.model.parameters(): # Freeze all the layers of the backbone
            param.requires_grad = False
        
    def forward(self, x):
        return self.model(x)

# Define the complete model
class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super(VisualFeatureExtractor, self).__init__()
        self.feature_extractor = ViTModel()
        
    def forward(self, x):
        features = [self.feature_extractor(part) for part in x]
        return(torch.cat(features, dim=1))

class CLIPModel(nn.Module):
  def __init__(self):
    super(CLIPModel, self).__init__()

    self.model, _ = clip.load("ViT-B/32")

    for param in self.model.parameters():
      param.requires_grad = False

  def forward(self, inputs):
    text_features = self.model.encode_text(inputs).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

# Define neural network architecture
class TextualFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextualFeatureExtractor, self).__init__()

        self.feature_extractor = CLIPModel()

    def forward(self, inputs):
        x1, x2, x3, x4, x5 = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :], inputs[:, 3, :], inputs[:, 4, :]

        x1 = self.feature_extractor(x1)

        x2 = self.feature_extractor(x2)

        x3 = self.feature_extractor(x3)

        x4 = self.feature_extractor(x4)

        x5 = self.feature_extractor(x5)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        return x

class AttributePredictor(nn.Module):
    def __init__(self, model_name='vit_small_patch32_224.augreg_in21k_ft_in1k'):
        super(AttributePredictor, self).__init__()
        self.cnn_model = timm.create_model(model_name, pretrained=True)
        self.cnn_model.reset_classifier(0)  # Remove the classification layer
        for param in self.cnn_model.parameters(): # Freeze all the layers of the backbone
            param.requires_grad = False

        self.num_features = self.cnn_model.num_features
        self.D1 = nn.Linear(self.num_features, 512)
        self.BN_1 = nn.BatchNorm1d(512)
        self.D2 = nn.Linear(512, 128)
        self.BN_2 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 170)  # Output 170-long boolean vector

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        
        x = self.D1(x)
        x = self.BN_1(x)
        x = nn.functional.leaky_relu(x)
        
        x = self.D2(x)
        x = self.BN_2(x)
        x = nn.functional.leaky_relu(x)
        
        out = self.out(x)
        
        # Apply sigmoid activation to produce boolean values
        out = torch.sigmoid(out)
        
        return out

class AttributeModel(nn.Module):
    def __init__(self, weights_path='bce_model_v2.pth'):
        super(AttributeModel, self).__init__()
        self.predictor = AttributePredictor()
        self.predictor.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        attributes = [self.predictor(part) for part in x]
        return(torch.cat(attributes, dim=1))