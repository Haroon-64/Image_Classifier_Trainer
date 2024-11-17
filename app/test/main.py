from fastapi import FastAPI
from pydantic import BaseModel
from config import Default_settings as settings
import data_load
import model_build
# import visualiser

app = FastAPI()

# Model to receive config data from Flutter
class Config(BaseModel):
  data_path: str
  model_size: str
  image_size: int
  transform: str = None
  num_classes: int
  epochs: int
  batch_size: int
  learning_rate: float
  output_path: str  


@app.get("/config", response_model=Config)
def get_config():
  return current_config

@app.post("/config")
def update_config(new_config: Config):
  global current_config
  current_config = new_config
  return {"message": "Configuration updated successfully", "config": current_config}

app.post("/data")
app.post("/model")

@app.post("/train")
def train():
    images, labels = data_load.load_data(settings['data_path'])
    model = model_build.build_model(settings['model_size'])
    model_build.train_model(model, images, labels)
    model.save("models/trained_model.h5")
    return {"status": "Training completed"}

@app.get("/status")
def status():
   pass

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) # web server for the API
