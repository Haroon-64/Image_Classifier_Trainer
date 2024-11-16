from fastapi import FastAPI
from pydantic import BaseModel
from config import settings
import data_load
import model_build
import visualiser

app = FastAPI()

# Model to receive config data from Flutter
class Config(BaseModel):
    data_path: str
    model_size: str
    epochs: int
    batch_size: int

@app.post("/configure")
def configure(config: Config):
    settings['data_path'] = config.data_path
    settings['model_size'] = config.model_size
    settings['epochs'] = config.epochs
    settings['batch_size'] = config.batch_size
    
    return {"status": "Configuration updated"}

@app.post("/train")
def train():
    images, labels = data_load.load_data(settings['data_path'])
    model = model_build.build_model(settings['model_size'])
    model_build.train_model(model, images, labels)
    model.save("models/trained_model.h5")
    return {"status": "Training completed"}

@app.get("/status")
def status():
    # Idealy, return training progress or model status
    return {"status": "Idle"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
