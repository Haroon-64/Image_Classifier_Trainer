import React, { useState } from "react";
import axios from "axios";
import "./App.css";

// Extend the Window interface to include the electron property
declare global {
  interface Window {
    electron?: {
      selectFolder: () => Promise<string>;
    };
  }
}

// Define TypeScript types for the configuration
interface Config {
  data_path: string;
  model_name: string;
  model_size: string;
  image_size: number;
  transform: string;
  num_classes: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  output_path: string;
  model?: any;
}

const App: React.FC = () => {
  const [config, setConfig] = useState<Config>({
    data_path: "",
    model_name: "resnet",
    model_size: "small",
    image_size: 224,
    transform: "None",
    num_classes: 2,
    epochs: 1,
    batch_size: 32,
    learning_rate: 0.001,
    output_path: "./output",
  });

  const [status, setStatus] = useState<string>("");
  const [modelTrained, setModelTrained] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<string>("");

  const pollTrainingProgress = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/train/progress");
      setTrainingProgress(`Training in progress: Epoch ${response.data.epoch}`);
    } catch (error) {
      setStatus(`Error: ${error}`);
    }
  };

  const updateConfig = async (newConfig: Config) => {
    try {
      await axios.post("http://127.0.0.1:8000/config", newConfig);
      setConfig(newConfig);
      setStatus("Configuration updated successfully.");
    } catch (error) {
      setStatus(`Error: ${error}`);
    }
  };

  const loadData = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/data");
      setStatus(response.data.message || "Data loaded successfully.");
    } catch (error) {
      setStatus(`Error: ${error}`);
    }
  };

  const startTraining = async () => {
    try {
      // Start training
      const response = await axios.post("http://127.0.0.1:8000/train");
      setStatus(response.data.message || "Training started.");
      
      // Poll progress
      const interval = setInterval(async () => {
        try {
          const progressResponse = await axios.get("http://127.0.0.1:8000/train/progress");
          const { epoch, total_epochs, status } = progressResponse.data;
          
          // Update progress
          setStatus(`Epoch ${epoch}/${total_epochs}: ${status}`);
          
          // Stop polling when training is complete
          if (epoch >= total_epochs || status.includes("complete")) {
            clearInterval(interval);
            setModelTrained(true);
            setStatus("Training completed.");
          }
        } catch (pollError) {
          console.error("Error fetching training progress:", pollError);
          setStatus("Error fetching training progress.");
          clearInterval(interval);
        }
      }, 1000);
    } catch (error) {
      console.error("Error starting training:", error);
      setStatus("Error starting training. Check backend logs for details.");
    }
  };

  const buildModel = async (pretrained: boolean = true): Promise<void> => {
    try {
      const response = await axios.post<{ message: string }>("http://127.0.0.1:8000/model", {
        model_name: config.model_name,
        model_size: config.model_size,
        pretr: pretrained,
      });
      setStatus(response.data.message || "Model built successfully.");
    } catch (error: unknown) {
      if (axios.isAxiosError(error) && error.response) {
        console.error("Error building model:", error.response.data);
        setStatus(`Error: ${error.response.data.error || error.message}`);
      } else {
        console.error("Unexpected error:", error);
        setStatus("An unexpected error occurred.");
      }
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setConfig((prevConfig) => ({
      ...prevConfig,
      [name]: value,
    }));
  };

  const saveModel = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/model/save");
      setStatus(response.data.message || "Model saved successfully.");
    } catch (error) {
      setStatus(`Error: ${error}`);
    }
  };

  const loadModel = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/model/load");
      setStatus(response.data.message || "Model loaded successfully.");
    } catch (error) {
      setStatus(`Error: ${error}`);
    }
  };

  const selectFolder = async () => {
    console.log("Selecting folder...");
    if (window.electron?.selectFolder) {
      console.log("Selecting folder...");
      const folderPath = await window.electron.selectFolder();
      if (folderPath) {
        setConfig((prevConfig) => ({
          ...prevConfig,
          data_path: folderPath,
        }));
        setStatus(`Folder selected: ${folderPath}`);
      } else {
        setStatus("Folder selection canceled.");
      }
    } else {
      setStatus("Folder picker not available.");
    }
  };

  return (
    <div className="App">
      <h1>FastAPI Image Classifier UI</h1>

      <div>
        <h2>Configuration</h2>
        <label>
          Data Path:
          <input
            type="text"
            placeholder="Paste or type path here"
            value={config.data_path}
            onChange={(e) => {
              console.log("Data Path:", e.target.value);
              setConfig((prevConfig) => ({
                ...prevConfig,
                data_path: e.target.value,
              }));
            }}
          />
        </label>

        {/* <button onClick={selectFolder}>Select Directory</button> */}

        <label>
          Model Name:
          <select
            name="model_name"
            value={config.model_name}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange(e)}
          >
            <option value="resnet">ResNet</option>
            <option value="mobilenet">MobileNet</option>
          </select>
        </label>

        <label>
          Model Size:
          <select
            name="model_size"
            value={config.model_size}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange(e)}
          >
            {config.model_name === "resnet" && (
              <>
                <option value="small">Small</option>
                <option value="medium">Medium</option>
                <option value="large">Large</option>
                <option value="xlarge">X-Large</option>
                <option value="xxlarge">XX-Large</option>
              </>
            )}
            {config.model_name === "mobilenet" && (
              <>
                <option value="small">Small</option>
                <option value="large">Large</option>
              </>
            )}
          </select>
        </label>

        <label>
          Image Size:
          <input
            type="number"
            name="image_size"
            value={config.image_size}
            onChange={handleInputChange}
          />
        </label>

        <label>
          Number of Classes:
          <input
            type="number"
            name="num_classes"
            value={config.num_classes}
            onChange={handleInputChange}
          />
        </label>

        <label>
          Epochs:
          <input
            type="number"
            name="epochs"
            value={config.epochs}
            onChange={handleInputChange}
          />
        </label>

        <label>
          Batch Size:
          <input
            type="number"
            name="batch_size"
            value={config.batch_size}
            onChange={handleInputChange}
          />
        </label>

        <label>
          Learning Rate:
          <input
            type="number"
            name="learning_rate"
            value={config.learning_rate}
            onChange={handleInputChange}
          />
        </label>

        <label>
          Output Path:
          <input
            type="text"
            name="output_path"
            value={config.output_path}
            onChange={handleInputChange}
          />
        </label>

        <button onClick={() => updateConfig(config)}>Update Config</button>
      </div>

      <div>
        <h2>Status</h2>
        <p>{status}</p>
      </div>

      <div>
        <button onClick={loadData}>Load Data</button>
        <button onClick={() => buildModel(true)}>Build Pretrained Model</button>
        <button onClick={() => buildModel(false)}>Build Non-Pretrained Model</button>
        <button onClick={startTraining}>Start Training</button>

        <div>
          <h2>Training Progress</h2>
          <p>{trainingProgress}</p>
        </div>
      </div>
      {modelTrained && (
        <div>
          <button onClick={saveModel}>Save Model</button>
          <button onClick={loadModel}>Load Model</button>
        </div>
      )}
    </div>
  );
};

export default App;