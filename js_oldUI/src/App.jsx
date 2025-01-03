import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const App = () => {
  const [config, setConfig] = useState({
    data_path: "",
    model_name: "resnet",
    model_size: "small",
    image_size: 224,
    transform: null,
    num_classes: 2,
    epochs: 1,
    batch_size: 32,
    learning_rate: 0.001,
    output_path: "./output",
  });

  const [status, setStatus] = useState("");

  const updateConfig = async (newConfig) => {
    try {
      await axios.post("http://127.0.0.1:8000/config", newConfig);
      setConfig(newConfig);
      setStatus("Configuration updated successfully.");
    } catch (error) {
      console.log(error.message)
      setStatus(`Error: ${error.message}`);
    }
  };

  const loadData = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/data");
      setStatus(response.data.message || "Data loaded successfully.");
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  const startTraining = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/train");
      setStatus(response.data.message || "Training started.");
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setConfig((prevConfig) => ({
      ...prevConfig,
      [name]: value,
    }));
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
            name="data_path"
            value={config.data_path}
            onChange={handleInputChange}
          />
        </label>
        <label>
          Model Size:
          <select
            name="model_size"
            value={config.model_size}
            onChange={handleInputChange}
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
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
        <button onClick={startTraining}>Start Training</button>
      </div>
    </div>
  );
};

export default App;
