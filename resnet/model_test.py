import torch
from torchvision import transforms
from PIL import Image
from resnet import ResNet, Bottleneck 

class_labels = ['Glioma','Meningioma','No_Tumor','Pituitary']


def load_model(weights_path):
    model = ResNet(Bottleneck, [3, 4, 6, 3])  # Adjust the architecture if needed
    model.load_state_dict(torch.load(weights_path,weights_only=True))
    model.eval()
    return model

def preprocess_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return input_batch


def test_model(image_path, weights_path):
    model = load_model(weights_path)
    input_batch = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_batch)
        output_class = torch.argmax(output,dim=1).sum().item()
        print(class_labels[output_class])

if __name__ == '__main__':
    image_path = r'data\MRI_classification\Testing\pituitary\Te-pi_0014.jpg'
    weights_path = 'Model\ResNetT50'  
    test_model(image_path, weights_path)

