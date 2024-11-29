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

