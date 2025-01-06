import torch
import torch.nn as nn
import torchvision.models as models

# Define a simple emotion model
class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate and convert the model
model = EmotionNet()
model.eval()

# Dummy input for ONNX export
dummy_input = torch.randn(1, 3, 224, 224)
onnx_file = "emotion_model.onnx"
torch.onnx.export(model, dummy_input, onnx_file,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
