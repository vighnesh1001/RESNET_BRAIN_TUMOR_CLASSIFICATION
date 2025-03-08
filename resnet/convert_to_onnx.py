import torch
from resnet import ResNet, Bottleneck

def load_model(weights_path):
    model = ResNet(Bottleneck, [3, 4, 6, 3])  # Adjust the architecture if needed
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model

def convert_to_onnx(weights_path, onnx_path):
    model = load_model(weights_path)
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the input size if needed
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
    print(f"Model has been converted to ONNX and saved at {onnx_path}")

if __name__ == '__main__':
    weights_path = 'Model\ResNetT50'  
    onnx_path = 'Model\model.onnx'  
    convert_to_onnx(weights_path, onnx_path)