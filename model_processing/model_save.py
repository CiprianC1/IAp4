import torch
import torch.nn as nn
import torchvision.models as models

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super(EmotionClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

model = EmotionClassifier()
model.load_state_dict(torch.load('best_model_augmentation.pkl', map_location=torch.device('cpu')))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
onnx_file = "emotion_model-8classes.onnx"
torch.onnx.export(model, dummy_input, onnx_file,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
