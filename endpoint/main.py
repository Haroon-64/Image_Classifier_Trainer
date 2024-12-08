from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mobilenet_v3_small, mobilenet_v3_large,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights, MobileNet_V2_Weights, MobileNet_V3_Large_Weights
)
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

# Global state
state = {
    "config": None,
    "model": None,
    "dataloader": None,
    "training_status": "Idle",
    "current_epoch": 0,
}


MODEL_MAP = {
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
    },
}
WEIGHTS_MAP = {
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
    },
}

class Config(BaseModel):
    data_path: str = Field(..., description="Path to the dataset folder", example="./data/images")
    model_name: str = Field(..., description="Model type ('resnet' or 'mobilenet')", example="resnet")
    model_size: str = Field(..., description="Model size ('small', 'medium', etc.)", example="small")
    image_size: int = Field(..., description="Size of input images", example=224)
    transform: str = Field(None, description="Data augmentation strategy", example="augmentation")
    num_classes: int = Field(..., description="Number of output classes", example=10)
    epochs: int = Field(..., description="Number of epochs for training", example=10)
    batch_size: int = Field(..., description="Batch size for training", example=32)
    learning_rate: float = Field(..., description="Learning rate for training", example=0.001)
    output_path: str = Field(..., description="Path to save models and outputs", example="./output")

@app.get("/")
def root():
    return {"message": "Welcome to the Model Training API"}

@app.post("/config", response_model=Config)
def update_config(config: Config):
    if not os.path.exists(config.data_path):
        return {"error": "Invalid data path"}
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    state["config"] = config
    return config

@app.post("/data")
def load_data():
    config = state["config"]
    if not config:
        return {"error": "Configuration not set"}
    if not os.path.exists(config.data_path):
        return {"error": "Invalid data path"}
    transform_list = [transforms.Resize((config.image_size, config.image_size))]
    if config.transform == "augmentation":
        transform_list.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)])
    transform_list.append(transforms.ToTensor())
    try:
        dataset = ImageFolder(root=config.data_path, transform=transforms.Compose(transform_list))
        state["dataloader"] = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        return {
            "message": "Data loaded successfully",
            "num_samples": len(dataset),
            "classes": dataset.classes,
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/model")
def build_model():
    config = state["config"]
    if not config:
        return {"error": "Configuration not set"}
    try:
        model_class = MODEL_MAP[config.model_name][config.model_size]
        weights = WEIGHTS_MAP[config.model_name].get(config.model_size)
        model = model_class(weights=weights)
        if config.model_name == "resnet":
            model.fc = torch.nn.Linear(model.fc.in_features, config.num_classes)
        elif config.model_name == "mobilenet":
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, config.num_classes)
        state["model"] = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return {"message": "Model built successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/model/save")
def save_model(model_path: str = None):
    """
    Save the current model to the specified path or the default path.
    """
    config = state["config"]
    model = state["model"]
    if not model:
        return {"error": "No model to save"}
    model_path = model_path or os.path.join(config.output_path, "model.pth")
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return {"message": f"Model saved successfully at {model_path}"}
    except Exception as e:
        return {"error": f"Failed to save model: {str(e)}"}

@app.post("/model/load")
def load_model(model_path: str = None):
    """
    Load a model from the specified path or the default path.
    """
    config = state["config"]
    if not config:
        return {"error": "Configuration not set"}
    model_path = model_path or os.path.join(config.output_path, "model.pth")
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}
    try:
        # Rebuild the model architecture
        model_class = MODEL_MAP[config.model_name][config.model_size]
        model = model_class()
        # Load model weights
        model.load_state_dict(torch.load(model_path,map_location=('cuda' if torch.cuda.is_available() else 'cpu')))
        state["model"] = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return {"message": f"Model loaded successfully from {model_path}"}
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

@app.post("/train")
def train(background_tasks: BackgroundTasks):
    """
    Start the training process in the background.
    """
    config = state["config"]
    model = state["model"]
    dataloader = state["dataloader"]
    if not model:
        return {"error": "No model initialized"}
    if not dataloader:
        return {"error": "No data loaded"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    def train_task():
        state["training_status"] = "Training in progress"
        try:
            for epoch in range(config.epochs):
                state["current_epoch"] = epoch + 1
                epoch_loss = 0.0
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {state['current_epoch']}/{config.epochs}, Loss: {avg_loss}")
            state["training_status"] = "Training complete"
        except Exception as e:
            state["training_status"] = f"Error during training: {str(e)}"

    background_tasks.add_task(train_task)
    return {"message": "Training started"}

@app.get("/train/progress")
def get_training_progress():
    """
    Get the current progress of training.
    """
    return {
        "epoch": state["current_epoch"],
        "current_epochs": state.get("current_epoch", 0),
        "status": state["training_status"],
    }

@app.post("/saliency")
def generate_saliency(image_path: str, target_class: int = 0):
    """
    Generate and save a saliency map for a given input image.
    """
    config = state["config"]
    model = state["model"]
    if not model:
        return {"error": "Model not built or loaded"}
    if not os.path.exists(image_path):
        return {"error": f"Image path '{image_path}' not found"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    try:
        saliency = Saliency(model)
        saliency_map = saliency.attribute(input_image, target=target_class).squeeze().cpu().numpy()
        plt.imshow(saliency_map, cmap="hot")
        saliency_path = os.path.join(config.output_path, "saliency_map.png")
        os.makedirs(config.output_path, exist_ok=True)
        plt.axis("off")
        plt.savefig(saliency_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return {"message": "Saliency map generated", "path": saliency_path}
    except Exception as e:
        return {"error": f"Failed to generate saliency map: {str(e)}"}

@app.post("/inference")
def inference(image_path: str):
    """
    Perform inference on a given image and return the predicted class.
    """
    config = state["config"]
    model = state["model"]
    if not model:
        return {"error": "Model not built or loaded"}
    if not os.path.exists(image_path):
        return {"error": f"Image file '{image_path}' not found"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            outputs = model(input_image)
            _, predicted_class = outputs.max(1)
        return {"predicted_class": predicted_class.item()}
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)