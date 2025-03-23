import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App: React.FC = () => {
  const [config, setConfig] = useState({
    data_path: '',
    model_name: 'resnet',
    model_size: 'small',
    image_size: 224,
    transform: 'none',
    num_classes: 2,
    epochs: 1,
    batch_size: 32,
    learning_rate: 0.001,
    output_path: './output',
    model_path: '',
  });

  const [trainingStatus, setTrainingStatus] = useState('Idle');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [inferenceResult, setInferenceResult] = useState('');
  const [saliencyPath, setSaliencyPath] = useState('');
  const [modelLoaded, setModelLoaded] = useState(false);
  
  const [notification, setNotification] = useState<string | null>(null);

  const showNotification = (message: string) => {
    setNotification(message);
    setTimeout(() => setNotification(null), 5000); // Hide notification after 5 seconds
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setConfig((prev) => ({ ...prev, [name]: value }));
  };

  const updateConfig = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/config', config);
      showNotification(response.data.message);
      await buildModel(); 
    } catch (error) {
      console.error(error);
      showNotification('Error updating config');
    }
  };

  const buildModel = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/model', {
        model_name: config.model_name,
        modelsize: config.model_size,
        pretr: true,
      });
      showNotification(response.data.message || 'Model built successfully!');
    } catch (error) {
      console.error(error);
      showNotification('Error building model');
    }
  };

  const loadModel = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/model/load', {
        model_path: config.model_path || '',
      });
      showNotification(response.data.message || 'Model loaded successfully!');
      setModelLoaded(true); 
    } catch (error) {
      console.error(error);
      showNotification('Error loading model');
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
      showNotification('Error starting training');
    }
  };

  const loadData = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/data', {
        data_path: config.data_path,
        transform: config.transform,
      });
      showNotification(response.data.message || 'Data loaded successfully!');
    } catch (error) {
      console.error(error);
      showNotification('Error loading data');
    }
  };

  const saveModel = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/model/save', {
        output_path: config.output_path,
      });
      showNotification(response.data.message || 'Model saved successfully!');
    } catch (error) {
      console.error(error);
      showNotification('Error saving model');
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedImage(e.target.files[0]);
    }
  };

  const performInference = async () => {
    if (!selectedImage) {
      showNotification('Please upload an image first.');
      return;
    }
  
    if (!modelLoaded) {
      showNotification('please load a model first.');
      return;
    }
  
    const formData = new FormData();
    formData.append('image', selectedImage); // Match the parameter name in the backend
  
    try {
      const { data } = await axios.post('http://127.0.0.1:8000/inference', formData, {
        headers: {
          'Content-Type': 'multipart/form-data', // Required for file uploads
        },
      });
      setInferenceResult(`Predicted Class: ${data.predicted_class}`);
    } catch (error) {
      console.error(error);
      showNotification('Error during inference');
    }
  };
  

  const generateSaliencyMap = async () => {
    if (!selectedImage) {
      showNotification('Please upload an image first.');
      return;
    }
  
    if (!modelLoaded) {
      showNotification('Please load a model first.');
      return;
    }
  
    const formData = new FormData();
    formData.append('image', selectedImage); // Match the parameter name in the backend
    try {
      const { data } = await axios.post('http://127.0.0.1:8000/saliency', formData, {
        headers: {
          'Content-Type': 'multipart/form-data', // Required for file uploads
        },
      });
      setSaliencyPath(data.path);
      showNotification('Saliency map generated successfully!');
    } catch (error) {
      console.error(error);
      showNotification('Error generating saliency map');
    }
  };
  

  return (
    <div className="App">
      {notification && <div className="notification">{notification}</div>}

      <div className="left-panel">
        <h2>Model Configuration</h2>
        <label>Data Path: <input name="data_path" value={config.data_path} onChange={handleInputChange} /></label>
        <label>Model Path: <input name="model_path" value={config.model_path} onChange={handleInputChange} /></label>

        <label>Model Name:
          <select name="model_name" value={config.model_name} onChange={handleInputChange}>
            <option value="resnet">ResNet</option>
            <option value="mobilenet">MobileNet</option>
          </select>
        </label>
        <label>Model Size: 
          <select name="modelsize" value={config.model_size} onChange={handleInputChange}>
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
        
        <button onClick={saveModel}>Save Model</button>
        <button onClick={startTraining}>Start Training</button>
        <p>{trainingStatus}</p>
      </div>

      <div className="right-panel">
        <h2>Inference and Saliency</h2>
        <input type="file" onChange={handleImageUpload} />
        
        <button onClick={loadModel}>Load Model for Inference</button> {/* Load Model Button */}
        <button onClick={performInference}>Perform Inference</button>
        <p>{inferenceResult}</p>
        <button onClick={generateSaliencyMap}>Generate Saliency Map</button>
        {saliencyPath && <img src={`http://127.0.0.1:8000/${saliencyPath}`} alt="Saliency Map" />}
      </div>
    </div>
  );
};

export default App;

