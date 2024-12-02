from re import M
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import os
from torchvision import transforms
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large  
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Large_Weights

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from captum.attr import Saliency
import matplotlib.pyplot as plt

from PIL import Image

app = FastAPI()


app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allow all origins for development
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

class Config(BaseModel):
  data_path: str = Field(..., description="Path to the dataset folder", example="./data/images")
  model_size: str = Field(..., description="Model size: 'small', 'medium', 'large', etc.", example="small")
  image_size: int = Field(..., description="Size of the input images (square dimensions)", example=224)
  transform: str = Field(None, description="Data augmentation strategy: 'augmentation' or None", example="augmentation")
  num_classes: int = Field(..., description="Number of output classes", example=10)
  epochs: int = Field(..., description="Number of training epochs", example=10)
  batch_size: int = Field(..., description="Batch size for training", example=32)
  learning_rate: float = Field(..., description="Learning rate for the optimizer", example=0.001)
  output_path: str = Field(..., description="Directory to save trained models and outputs", example="./output")

current_config = Config(
  data_path=r"",
  model_size="small",
  image_size=224,
  num_classes=2,
  epochs=1,
  batch_size=32,
  learning_rate=0.001,
  output_path="./output",
)
current_model = None
current_epoch = 0


training_status = "Idle"
current_dataloader = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.get("/")
def read_root():
    return {"message": "root endpoint"}

@app.get("/config", response_model=Config)
def get_config():
  return current_config

@app.post("/config")
async def update_config(config: Config):
    """
    Update the current configuration with the provided values.
    Args:
        config (Config): The new configuration values.
    Returns:
        dict: A message indicating the status of the configuration update.
    """
    global current_config
    if not os.path.exists(config.data_path):
        print(f"Invalid data path: {config.data_path}")
        return {"message": f"Error: Data path {config.data_path} does not exist."}
    
    current_config = config
    print(f"Configuration updated: {current_config}")
    return {"message": "Configuration updated successfully."}



@app.post("/data")
def load_data():
  """
  Load the dataset from the specified data path and return metadata.
  """

  global current_config, current_dataloader
  print(f"Loading data from path: {current_config.data_path}")
  # Check if the data path exists
  if not os.path.exists(current_config.data_path):
    return {"error": "Invalid data path"}

  # Define transformations
  transform_list = [transforms.Resize((current_config.image_size, current_config.image_size))]
  if current_config.transform == "augmentation":
    transform_list += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
  transform_list += [transforms.ToTensor()]
  transform = transforms.Compose(transform_list)

  try:
    # Load the dataset
    dataset = ImageFolder(root=current_config.data_path, transform=transform)
    current_dataloader = DataLoader(dataset, batch_size=current_config.batch_size, shuffle=True)

    # Return metadata
    print(f"Data loaded successfully: {len(dataset)} samples found.")
    return {
      "message": "Data loaded successfully",
      "num_samples": len(dataset),
      "num_classes": len(dataset.classes),
      "classes": dataset.classes,
    }

  except Exception as e:
    return {"error": str(e)}



@app.post("/model")
def build_model(model_name: str = "resnet", model_size: str = "small", pretr: bool = True):
    """
    Build a model based on the specified model name and size.
    """
    global current_model, current_config

    # Mapping model names and sizes to model classes
    model_map = {
        "resnet": {
            "small": resnet18,
            "medium": resnet34,
            "large": resnet50,
            "xlarge": resnet101,
            "xxlarge": resnet152,
        },
        "mobilenet": {
            "small": mobilenet_v3_small,
            "large": mobilenet_v3_large,
        }
    }

    # Ensure the model name and size are valid
    if model_name not in model_map or model_size not in model_map[model_name]:
        error_message = f"Invalid model name or size: {model_name} {model_size}"
        print(error_message)
        return {"error": error_message}

    try:
        # Select the appropriate model class and load weights
        model_class = model_map[model_name][model_size]
        weights = None
        if pretr:
            weights_map = {
                "resnet": {
                    "small": ResNet18_Weights.IMAGENET1K_V1,
                    "medium": ResNet34_Weights.IMAGENET1K_V1,
                    "large": ResNet50_Weights.IMAGENET1K_V1,
                    "xlarge": ResNet101_Weights.IMAGENET1K_V1,
                    "xxlarge": ResNet152_Weights.IMAGENET1K_V1,
                },
                "mobilenet": {
                    "small": MobileNet_V2_Weights.IMAGENET1K_V1,
                    "large": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                }
            }
            weights = weights_map[model_name].get(model_size, None)

        # Initialize the model
        current_model = model_class(weights=weights)

        # Adjust the final layer to match the number of classes
        if model_name == "resnet":
            current_model.fc = torch.nn.Linear(current_model.fc.in_features, current_config.num_classes)
        elif model_name == "mobilenet":
            current_model.classifier[1] = torch.nn.Linear(current_model.classifier[1].in_features, current_config.num_classes)

        # Move the model to the selected device
        current_model = current_model.to(device)

        print(f"Model '{model_name} {model_size}' built successfully.")
        return {"message": f"Model '{model_name} {model_size}' built successfully"}

    except Exception as e:
        error_message = f"Failed to build model: {str(e)}"
        print(error_message)
        return {"error": error_message}


  
@app.post("/model/save")
def save_model():
  """
  Save the current model to the output path specified in the configuration.
  """
  global current_config
  if current_model is None:
    return {"error": "No model to save"}

  model_path = f"{current_config.output_path}/model.pth"
  print(f"Saving model to: {model_path}")
  os.makedirs(current_config.output_path, exist_ok=True)
  torch.save(current_model.state_dict(), model_path)

  return {"message": f"Model saved successfully at {model_path}"}

@app.post("/model/load")
def load_model():
  """
  Load a model from the output path specified in the configuration.
  """
  global current_config
  model_path = f"{current_config.output_path}/model.pth"
  
  if not os.path.exists(model_path):
    return {"error": "Model file not found"}
  
  model_map = {
    "small": resnet18,
    "medium": resnet34,
    "large": resnet50,
    "xlarge": resnet101,
    "xxlarge": resnet152,
  }

  if current_config.model_size in model_map:
    model_class = model_map[current_config.model_size]
    current_model = model_class(pretrained=False)  # Don't load pre-trained weights here
    current_model.load_state_dict(torch.load(model_path))
    current_model.eval()
    print(f"Model loaded successfully from: {model_path}, size: {current_config.model_size}, classes: {current_config.num_classes}")
    return {"message": "Model loaded successfully"}
  else:
    return {"error": "Invalid model size"}

@app.post("/train")

def train(background_tasks: BackgroundTasks):
    """
    Endpoint to start the training process.
    This function initiates the training of a machine learning model using the current configuration, model, and dataloader.
    It runs the training loop for a specified number of epochs and updates the training status accordingly.
    Args:
      background_tasks (BackgroundTasks): FastAPI BackgroundTasks instance to handle background tasks.
    Returns:
      dict: A dictionary containing the status of the training process. If an error occurs, it returns an error message.
    """
    global current_model, current_dataloader, current_config, training_status, current_epoch
    print(current_model,type(current_model))
    if not current_model:
        training_status = "Error: Model not built"
        return {"error": training_status}

    if current_dataloader is None:
        training_status = "Error: DataLoader not initialized"
        return {"error": training_status}

    # Move model to device
    current_model = current_model.to(device)

    # Set loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(current_model.parameters(), lr=current_config.learning_rate)

    training_status = "Training in progress"
    try:
        for epoch in range(current_config.epochs):
            current_epoch = epoch + 1
            epoch_loss = 0.0
            for images, labels in current_dataloader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = current_model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(current_dataloader)
            print(f"Epoch {current_epoch}/{current_config.epochs}, Loss: {avg_loss}")

        training_status = "Training complete"
    except Exception as e:
        training_status = f"Error during training: {str(e)}"
        return {"error": training_status}

    return {"message": "Training complete"}

@app.get("/train/progress")
async def get_progress():
    
    print(f"Progress request: Epoch {current_epoch}/{current_config.epochs}, Status: {training_status}")
    return {"epoch": current_epoch, "total_epochs": current_config.epochs, "status": training_status}


@app.get("/status")
def status():
  global training_status
  return {"status": training_status}

@app.post("/saliency")
def generate_saliency(image_path: str, target_class: int = 0):
  global current_config

  if current_model is None:
    return {"error": "Model not built yet"}
  
  if not os.path.exists(image_path):
    return {"error": f"Image path '{image_path}' not found"}
  
  # Prepare image
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  image = Image.open(image_path).convert("RGB")
  transform = transforms.Compose([
    transforms.Resize((current_config.image_size, current_config.image_size)),
    transforms.ToTensor(),
  ])
  input_image = transform(image).unsqueeze(0).to(device)
  
  # Saliency map generation
  saliency = Saliency(current_config.model)
  saliency_map = saliency.attribute(input_image, target=target_class).squeeze().cpu().numpy()
  
  # Visualize and save
  plt.imshow(saliency_map, cmap='hot')
  saliency_path = os.path.join(current_config.output_path, "saliency_map.png")
  plt.axis('off')
  os.makedirs(current_config.output_path, exist_ok=True)
  plt.savefig(saliency_path, bbox_inches='tight', pad_inches=0)
  plt.close()

  return {"message": "Saliency map generated", "path": saliency_path}

@app.post("/inference")
def inference(image_path: str):
  """
  Perform inference on a single image and return the predicted class.
  """
  global current_config, current_model

  if current_config.model is None:
    return {"error": "Model not built or loaded"}

  # Ensure the model is in evaluation mode
  current_model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  current_model = current_model.to(device)

  try:
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
      transforms.Resize((current_config.image_size, current_config.image_size)),
      transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
      outputs = current_model(input_image)
      _, predicted_class = outputs.max(1)
    
    return {"predicted_class": predicted_class.item()}
  except Exception as e:
    return {"error": f"Inference failed: {str(e)}"}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)