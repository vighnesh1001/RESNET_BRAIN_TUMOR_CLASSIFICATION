import gradio as gr
from onnx_inference import infer_onnx_model
from gradio.themes import Glass


def predict(image):
    onnx_path = 'Model/model.onnx' 
    class_label = infer_onnx_model(image, onnx_path)
    return class_label

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload MRI Image"),
    outputs=gr.Textbox(label="Predicted Class"),
    title="MRI Classification of brain tumor using ResNet50",
    description="Upload an MRI image to classify it using a pre-trained model.",
    examples=[
        ["data/MRI_classification/Testing/notumor/Te-no_0398.jpg"],
        ["data/MRI_classification/Testing/glioma/Te-gl_0010.jpg"]
    ],
    theme="Glass",
    allow_flagging="manual"  # Enable manual flagging
)

if __name__ == '__main__':
    iface.launch()
