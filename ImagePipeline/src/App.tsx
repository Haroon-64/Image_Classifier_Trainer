import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App: React.FC = () => {
  const [config, setConfig] = useState({
    data_path: '',
    model_name: 'resnet',
    modelsize: 'small',
    image_size: 224,
    transform: 'none',
    num_classes: 2,
    epochs: 1,
    batch_size: 32,
    learning_rate: 0.001,
    output_path: './output',
  });

  const [trainingStatus, setTrainingStatus] = useState('Idle');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [inferenceResult, setInferenceResult] = useState('');
  const [saliencyPath, setSaliencyPath] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setConfig((prev) => ({ ...prev, [name]: value }));
  };

  const updateConfig = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/config', config);
      alert(response.data.message);
      await buildModel(); // Call build model after updating the config
    } catch (error) {
      console.error(error);
    }
  };

  const buildModel = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/model', {
        model_name: config.model_name,
        modelsize: config.modelsize,
        pretr: true, // Assuming we want to use pre-trained weights
      });
      alert(response.data.message || 'Model built successfully!');
    } catch (error) {
      console.error(error);
      alert('Error building model');
    }
  };

  const startTraining = async () => {
    try {
      await axios.post('http://127.0.0.1:8000/train');
      setTrainingStatus('Training in progress...');
      const interval = setInterval(async () => {
        const { data } = await axios.get('http://127.0.0.1:8000/train/progress');
        setTrainingStatus(`${data.status} (Epoch ${data.epoch}/${data.total_epochs})`);
        if (data.status === 'Training complete') clearInterval(interval);
      }, 5000);
    } catch (error) {
      console.error(error);
      alert('Error starting training');
    }
  };

  const loadData = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/data', {
        data_path: config.data_path,
        transform: config.transform,
      });
      alert(response.data.message || 'Data loaded successfully!');
    } catch (error) {
      console.error(error);
      alert('Error loading data');
    }
  };

  const loadModel = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/model/load', {
        output_path: config.output_path,
      });
      alert(response.data.message || 'Model loaded successfully!');
    } catch (error) {
      console.error(error);
      alert('Error loading model');
    }
  };

  const saveModel = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/model/save', {
        output_path: config.output_path,
      });
      alert(response.data.message || 'Model saved successfully!');
    } catch (error) {
      console.error(error);
      alert('Error saving model');
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedImage(e.target.files[0]);
    }
  };

  const performInference = async () => {
    if (!selectedImage) {
      alert('Please upload an image first.');
      return;
    }
    const formData = new FormData();
    formData.append('image_path', selectedImage);
    try {
      const { data } = await axios.post('http://127.0.0.1:8000/inference', formData);
      setInferenceResult(`Predicted Class: ${data.predicted_class}`);
    } catch (error) {
      console.error(error);
      alert('Error during inference');
    }
  };

  const generateSaliencyMap = async () => {
    if (!selectedImage) {
      alert('Please upload an image first.');
      return;
    }
    const formData = new FormData();
    formData.append('image_path', selectedImage);
    try {
      const { data } = await axios.post('http://127.0.0.1:8000/saliency', formData);
      setSaliencyPath(data.path);
    } catch (error) {
      console.error(error);
      alert('Error generating saliency map');
    }
  };

  return (
    <div className="App">
      <div className="left-panel">
        <h2>Model Configuration</h2>
        <label>Data Path: <input name="data_path" value={config.data_path} onChange={handleInputChange} /></label>
        <label>Model Name:
          <select name="model_name" value={config.model_name} onChange={handleInputChange}>
            <option value="resnet">ResNet</option>
            <option value="mobilenet">MobileNet</option>
          </select>
        </label>
        <label>Model Size: 
          <select name="modelsize" value={config.modelsize} onChange={handleInputChange}>
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
            <option value="xlarge">X-Large</option>
            <option value="xxlarge">XX-Large</option>
          </select>
        </label>
        <label>Image Size: <input name="image_size" type="number" value={config.image_size} onChange={handleInputChange} /></label>
        <label>Transform: 
          <select name="transform" value={config.transform} onChange={handleInputChange}>
            <option value="none">None</option>
            <option value="augmentation">Augmentation</option>
          </select>
        </label>
        <label>Number of Classes: <input name="num_classes" type="number" value={config.num_classes} onChange={handleInputChange} /></label>
        <label>Epochs: <input name="epochs" type="number" value={config.epochs} onChange={handleInputChange} /></label>
        <label>Batch Size: <input name="batch_size" type="number" value={config.batch_size} onChange={handleInputChange} /></label>
        <label>Learning Rate: <input name="learning_rate" type="number" value={config.learning_rate} step="0.0001" onChange={handleInputChange} /></label>
        <button onClick={updateConfig}>Update Config & Build Model</button>
        <button onClick={loadData}>Load Data</button>
        {/* <button onClick={loadModel}>Load Model</button> */}
        <button onClick={saveModel}>Save Model</button>
        <button onClick={startTraining}>Start Training</button>
        <p>{trainingStatus}</p>
      </div>

      <div className="right-panel">
        <h2>Inference and Saliency</h2>
        <input type="file" onChange={handleImageUpload} />
        <button onClick={performInference}>Perform Inference</button>
        <p>{inferenceResult}</p>
        <button onClick={generateSaliencyMap}>Generate Saliency Map</button>
        {saliencyPath && <img src={`http://127.0.0.1:8000/${saliencyPath}`} alt="Saliency Map" />}
      </div>
    </div>
  );
};

export default App;
