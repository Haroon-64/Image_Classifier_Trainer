from fastapi import FastAPI
from pydantic import BaseModel, Field

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.utils.tensorboard import SummaryWriter

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from captum.attr import Saliency
import matplotlib.pyplot as plt

from PIL import Image

app = FastAPI()


class Config(BaseModel):
  data_path: str = Field(..., description="Path to the dataset folder", example="./data/images")
  model: nn.Module = None
  model_size: str = Field(..., description="Model size: 'small' or 'large'", example="small")
  image_size: int = Field(..., description="Size of the input images (square dimensions)", example=224)
  transform: str = Field(None, description="Data augmentation strategy: 'augmentation' or None", example="augmentation")
  num_classes: int = Field(..., description="Number of output classes", example=10)
  epochs: int = Field(..., description="Number of training epochs", example=10)
  batch_size: int = Field(..., description="Batch size for training", example=32)
  learning_rate: float = Field(..., description="Learning rate for the optimizer", example=0.001)
  output_path: str = Field(..., description="Directory to save trained models and outputs", example="./output")


current_config = Config(
  data_path="",
  model_size="small",
  image_size=224,
  num_classes=2,
  epochs=1,
  batch_size=32,
  learning_rate=0.001,
  output_path="./output",
)

training_status = "Idle"
current_dataloader = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.get("/config", response_model=Config)
def get_config():
  return current_config

@app.post("/config")
def update_config(new_config: Config):
  global current_config
  current_config = new_config
  return {"message": "Configuration updated successfully"}

@app.post("/data")
def load_data():
  global current_config, current_dataloader

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
    return {
      "message": "Data loaded successfully",
      "num_samples": len(dataset),
      "num_classes": len(dataset.classes),
      "classes": dataset.classes,
    }

  except Exception as e:
    return {"error": str(e)}

@app.post("/model")
def build_model(pretr: bool = True):
  """
  Build and store the model in the configuration based on the model size.
  """
  global current_config

  model_map = {
    "small": resnet18,
    "medium": resnet34,
    "large": resnet50,
    "xlarge": resnet101,
    "xxlarge": resnet152,
  }

  if current_config.model_size in model_map:
    current_config.model = model_map[current_config.model_size](pretrained=pretr)
    return {"message": f"Model '{current_config.model_size}' built successfully"}
  elif current_config.model_size == "custom":
    return {"error": "Custom models not supported yet"}
  else:
    return {"error": "Invalid model size"}
  
@app.post("/model/save")
def save_model():
  """
  Save the current model to the output path specified in the configuration.
  """
  global current_config
  if current_config.model is None:
    return {"error": "No model to save"}
  
  model_path = f"{current_config.output_path}/model.pth"
  os.makedirs(current_config.output_path, exist_ok=True)
  
  # Save the model's state dictionary
  torch.save(current_config.model.state_dict(), model_path)
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
    current_config.model = model_class(pretrained=False)  # Do not load pretrained weights
    current_config.model.load_state_dict(torch.load(model_path))
    current_config.model.eval()  # Set model to evaluation mode
    return {"message": "Model loaded successfully"}
  else:
    return {"error": "Invalid model size"}

@app.post("/train")
def train():
  global current_config, current_dataloader, training_status

  if current_config.model is None:
    training_status = "Error: Model not built"
    return {"error": training_status}

  if current_dataloader is None:
    training_status = "Error: DataLoader not initialized"
    return {"error": training_status}

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  current_config.model = current_config.model.to(device)

  criterion = CrossEntropyLoss()
  optimizer = Adam(current_config.model.parameters(), lr=current_config.learning_rate)

  log_dir = os.path.join(current_config.output_path, "logs")
  os.makedirs(log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir)

  training_status = "Training in progress"
  try:
    for epoch in range(current_config.epochs):
      epoch_loss = 0.0
      for i, (images, labels) in enumerate(current_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = current_config.model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar("Loss/step", loss.item(), epoch * len(current_dataloader) + i)
      
      avg_loss = epoch_loss / len(current_dataloader)
      writer.add_scalar("Loss/epoch", avg_loss, epoch)
      print(f"Epoch {epoch+1}/{current_config.epochs}, Loss: {avg_loss}")
    
    training_status = "Training complete"
  except Exception as e:
    training_status = f"Error during training: {str(e)}"
    writer.close()
    return {"error": training_status}
  
  writer.close()
  return {"message": "Training complete"}

@app.get("/status")
def status():
  global training_status
  return {"status": training_status}

@app.post("/saliency")
def generate_saliency(image_path: str, target_class: int = 0):
  global current_config

  if current_config.model is None:
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
  global current_config

  if current_config.model is None:
    return {"error": "Model not built or loaded"}

  # Ensure the model is in evaluation mode
  current_config.model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  current_config.model = current_config.model.to(device)

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
      outputs = current_config.model(input_image)
      _, predicted_class = outputs.max(1)
    
    return {"predicted_class": predicted_class.item()}
  except Exception as e:
    return {"error": f"Inference failed: {str(e)}"}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)