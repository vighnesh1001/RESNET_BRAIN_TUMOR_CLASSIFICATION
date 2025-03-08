import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Define the list of class labels
class_labels = ['Glioma','Meningioma','No_Tumor','Pituitary']


def preprocess_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).numpy()  # Convert to numpy array
    return input_batch

def infer_onnx_model(image_path, onnx_path):
    session = ort.InferenceSession(onnx_path)
    input_batch = preprocess_image(image_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_batch})
    output_class = np.argmax(output[0], axis=1).item()
    class_label = class_labels[output_class]
    return class_label

if __name__ == '__main__':
    image_path = 'data\\MRI_classification\\Testing\\notumor\\Te-no_0398.jpg'  # Replace with the path to your input image
    onnx_path = 'Model\model.onnx'  # Path to your ONNX model
    class_label = infer_onnx_model(image_path, onnx_path)
    print(f'Predicted class: {class_label}')
    
    
    