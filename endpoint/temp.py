from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import torch

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
 protected_namespaces = (),
  data_path="",
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
  pass

@app.post("/config")
def set_config(config: Config):
  pass

@app.post("/config")
async def update_config(config: Config):
  pass

@app.post("/data")
def load_data():
  pass

@app.post("/model")
def build_model(pretr: bool = True):
  pass

@app.post("/model/save")
def save_model():
  pass

@app.post("/model/load")
def load_model():
  pass

@app.post("/train")
def train(background_tasks: BackgroundTasks):
  pass

@app.get("/train/progress")
async def get_progress():
  pass

@app.get("/status")
def status():
  pass

@app.post("/saliency")
def generate_saliency(image_path: str, target_class: int = 0):
  pass

@app.post("/inference")
def inference(image_path: str):
  pass

def build_model(model_name: str, model_size: str, pretr: bool = True):
    """
    Build a model based on the specified model name and size.
    """
    print("harron was here")
    global current_model, current_config

    # Supported models
    model_map = {
        "resnet": {
            "small": resnet18,
            "medium": resnet34,
            "large": resnet50,
            "xlarge": resnet101,
            "xxlarge": resnet152,
        },
        "mobilenet": {
            "small": mobilenet_v2,
            "large": mobilenet_v3_large,
        },
    }

    # Weights map for pretrained models
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
        },
    }

    if model_name not in model_map or model_size not in model_map[model_name]:
        error_message = f"Invalid model name or size: {model_name}, {model_size}"
        print(error_message)
        return {"error": error_message}

    try:
        print("in try build-model")
        # Select the model class and weights
        model_class = model_map[model_name][model_size]
        weights = None

        if pretr:
            weights = weights_map.get(model_name, {}).get(model_size)

        # Initialize the model
        current_model = model_class(weights=weights)
        print("after current model in build model")

        # Adjust the final layer to match the number of classes if using ResNet
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
        raise HTTPException(status_code=500, detail=error_message)
