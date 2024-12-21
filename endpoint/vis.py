from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v3_small, mobilenet_v3_large
import torch
from torchviz import make_dot
from PIL import Image
import io

MODEL_MAP = {
    "resnet": {
        "small": resnet18,
        "medium": resnet34,
        "large": resnet50,
    },
    "mobilenet": {
        "small": mobilenet_v3_small,
        "large": mobilenet_v3_large,
    },
}

def graph_model(model):
    """
    Generate the model's graph with layer names and sizes and return it as a PIL image.
    Args:
        model: The PyTorch model.
    Returns:
        PIL.Image: Image of the model graph.
    """
    dummy_input = torch.randn(1, 3, 224, 224)

    # Generate model's computation graph
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    
    # Convert the graph to a PNG image 
    with io.BytesIO() as img_buffer:
        graph.render(format='png', cleanup=True)  # Render to stdout
        img_buffer.write(graph.pipe(format='png'))  # Write the PNG image to the buffer
        img_buffer.seek(0)  # Rewind to the start of the buffer
        
        # Open the image from memory
        img = Image.open(img_buffer)
    
    return img

if __name__ == "__main__": 
    model = resnet18(pretrained=False)
    image = graph_model(model)
    image.show()

