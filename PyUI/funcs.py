import os
from typing import Literal
from torchvision import transforms
from torchvision.datasets import ImageFolder

from states  import MODEL_MAP, WEIGHTS_MAP, state

import torch
from torch.utils.data import DataLoader,random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from captum.attr import Saliency
import matplotlib.pyplot as plt
from torchviz import make_dot
from PIL import Image
import io

def load_data(data_path: str,
              image_size: int = 224,
              test_size: float = 0.2,
              transform: str = None,
              BATCHSIZE: int = 32,
              val_size: float = 0.1,  
              include_val: bool = False):# -> dict[str, str] | dict[str, DataLoader]:  
    
    if not os.path.exists(data_path):
        return {"error": "Invalid data path"}
    
    # Setup transformations
    transform_list = [transforms.Resize((image_size, image_size))]
    if transform == "augmentation":
        transform_list.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)])
    transform_list.append(transforms.ToTensor())
    
    try:
        # Load the full dataset
        full_dataset = ImageFolder(root=data_path, transform=transforms.Compose(transform_list))
        num_classes = len(full_dataset.classes)

        # Split into train and test datasets
        train_size = int((1 - test_size) * len(full_dataset))  
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Further split train dataset into train and validation datasets if include_val is True
        if include_val:
            val_size = int(val_size * train_size)
            train_size = train_size - val_size  # Adjust size of training dataset after split
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
            return {
                'train_loader': DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True),
                'test_loader': DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False),
                'val_loader': val_loader,
                'num_classes': num_classes
            }

        # If no validation is needed, return only train and test loaders
        return {
            'train_loader': DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True),
            'test_loader': DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False),
            'num_classes': num_classes
        }

    except Exception as e:
        return {"error": str(e)}    


def graph_model(model):
    """
    Generate the model's graph with layer names and sizes and return it as a PIL image.
    Args:
        model: The PyTorch model.
    Returns:
        PIL.Image: Image of the model graph.
    """
    dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)

    # Generate model's computation graph
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    
    # Convert the graph to a PNG image 
    img_buffer = io.BytesIO()
    graph.render(format='png', cleanup=True)  # Render to stdout
    img_buffer.write(graph.pipe(format='png'))  # Capture PNG image in memory
    img_buffer.seek(0)  # Rewind to the start of the buffer

    pil_image = Image.open(img_buffer)
    
    return pil_image


def build_model(model_name: Literal['resnet', 'mobilenet'], 
                model_size: str,
                num_classes: int,
                pretrained: bool = True):
    try:
        # Get the model class and weights
        model_class = MODEL_MAP[model_name][model_size]
        weights = WEIGHTS_MAP[model_name].get(model_size) if pretrained else None
        model = model_class(weights=weights)

        # Modify the final layer based on the model
        if model_name == "resnet":
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        elif model_name == "mobilenet":
            if model_size == "small":
                # For MobileNetV3 Small, replace the classifier 
                in_features = model.classifier[-1].in_features  # Get input features from the last Linear layer
                model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.2),  # Keep dropout layer consistent
                    torch.nn.Linear(in_features, num_classes)  # Replace final Linear layer
                )
            elif model_size == "large":
                # For MobileNetV3 Large, replace the last Linear layer of the classifier
                model.classifier[-1] = torch.nn.Linear(1280, num_classes)  # 1280 is the input size for V3 large

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return model
    except Exception as e:
        print(f"Error during model building: {e}")
        raise


     
    
def validate(model, dataloader, criterion):
    """
    Validate the model performance on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during validation
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def train(model,
          train_dataloader,
          val_dataloader=None,
          optimizer=None,
          epochs=2,
          criterion=CrossEntropyLoss(),
          start_learning_rate=1e-3):
    """
    Start the training process with optional validation.
    """

    # Validate arguments
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be an instance of torch.nn.Module")
    
    if not isinstance(train_dataloader, torch.utils.data.DataLoader):
        raise ValueError("train_dataloader must be an instance of torch.utils.data.DataLoader")
    
    if val_dataloader is not None and not isinstance(val_dataloader, torch.utils.data.DataLoader):
        raise ValueError("val_dataloader must be an instance of torch.utils.data.DataLoader or None")
    
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    
    if not isinstance(optimizer, (type(None), torch.optim.Optimizer)):
        raise ValueError("optimizer must be an instance of torch.optim.Optimizer or None")
    
    if not isinstance(criterion, torch.nn.Module):
        raise ValueError("criterion must be an instance of torch.nn.Module")
    
    if not isinstance(start_learning_rate, (float, int)) or start_learning_rate <= 0:
        raise ValueError("start_learning_rate must be a positive number")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=start_learning_rate)

    try:
        for epoch in range(epochs):
            state["current_epoch"] = epoch + 1
            epoch_loss = 0.0
            model.train()  # Set the model to training mode
            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {state['current_epoch']}/{epochs}, Training Loss: {avg_loss:.4f}")
            
            # Perform validation if val_dataloader is provided
            if val_dataloader:
                validate(model, val_dataloader, criterion)
            
        state["training_status"] = "Training complete"
        return model
    except Exception as e:
        state["training_status"] = "error"
        raise RuntimeError(f"Error during training: {str(e)}")

def save_model(model,
               model_path: str=None):
    
    """
    Save the current model to the specified path or the default path.
    """

    if not model:
        raise "no model to save"
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        raise {"error": f"Failed to save model: {str(e)}"}

def load_model(model_name,
               model_size,
               model_path: str=None,
               weights: Literal['pretrained', '']=None):
    """
    Load a model from the specified path or preexisting.
    """

    try:
        if model_path and not os.path.exists(model_path):
            raise Exception(f"Model file not found at {model_path}")
        
        # Rebuild the model architecture
        model_class = MODEL_MAP[model_name][model_size]
        model = model_class()

        if weights == 'pretrained':
            # Load pretrained weights
            model_weights = WEIGHTS_MAP[model_name][model_size]
            model.load_state_dict(model_weights)
        else:
            if not os.path.exists(weights):
                raise Exception(f"Model weights file not found at {weights}")
            model.load_state_dict(torch.load(weights, map_location=('cuda' if torch.cuda.is_available() else 'cpu')))
        return model

    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def inference(image_path: str,
              model,
              image_size=224):
    """
    Perform inference on a given image and return the predicted class.
    """

    if not os.path.exists(image_path):
        raise {"error": f"Image file '{image_path}' not found"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            outputs = model(input_image)
            _, predicted_class = outputs.max(1)
        return image, predicted_class.item()
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}
    
def generate_saliency(image_path: str,
                      model,
                      image_size=224,
                      output_path=None,
                    target_class: int=0):
    """
    Generate and save a saliency map for a given input image.
    """

    if not os.path.exists(image_path):
        return {"error": f"Image path '{image_path}' not found"}
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    input_image = transform(image).unsqueeze(0).to(device)

    try:
        saliency = Saliency(model)
        saliency_map = saliency.attribute(input_image, target=target_class).squeeze().cpu().numpy()

        # Reduce dimensions if necessary
        if len(saliency_map.shape) > 2:
            saliency_map = saliency_map.mean(axis=0)

        print(f"Saliency map shape: {saliency_map.shape}")



        # Save or return the saliency map
        if output_path:
            plt.savefig(output_path)
            print(f"Saliency map saved to {output_path}")
        else:
            plt.show()
        return saliency_map
        
    except Exception as e:
        raise {"error": f"Failed to generate saliency map: {str(e)}"}