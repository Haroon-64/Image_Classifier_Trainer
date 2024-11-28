import React, { useState } from "react";
import axios from "axios";
import "./App.css";

// Define TypeScript types for the configuration
interface Config {
    data_path: string;
    model_size: string;
    image_size: number;
    transform: string | null;
    num_classes: number;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    output_path: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    model?: any; // This is optional, since model is set dynamically in FastAPI
}


const App: React.FC = () => {
    const [config, setConfig] = useState<Config>({
        data_path: "",
        model_size: "small",
        image_size: 224,
        transform: "null",   // transforms not handled yet so set to null
        num_classes: 2,
        epochs: 1,
        batch_size: 32,
        learning_rate: 0.001,
        output_path: "./output",
    });

    const [status, setStatus] = useState<string>("");
    const [modelTrained, setModelTrained] = useState<boolean>(false); // Track training status
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
            // console.log(error)
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
            const response = await axios.post("http://127.0.0.1:8000/train");
            setStatus(response.data.message || "Training started.");
            const interval = setInterval(pollTrainingProgress, 1000); // Poll every second
            setTimeout(() => {
                clearInterval(interval);
                setModelTrained(true);
                setStatus("Training completed.");
            }, config.epochs * 1000);
        } catch (error) {
            setStatus(`Error: ${error}`);
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
                        onChange={(e) =>
                            setConfig((prevConfig) => ({
                                ...prevConfig,
                                data_path: e.target.value,
                            }))
                        }
                    />
                </label>

                <label>
                    Or select directory:
                    <input
                        type="file"
                        webkitdirectory="true"
                        onChange={(e) => {
                            const files = e.target.files;
                            if (files && files.length > 0) {
                                const directoryPath = files[0].webkitRelativePath.split("/")[0];
                                const fullPath = e.target.value; // Capture full path
                                setConfig((prevConfig) => ({
                                    ...prevConfig,
                                    data_path: fullPath, // Pass the full path to backend
                                }));
                                setStatus(`Directory selected: ${fullPath}`);
                            }
                        }}
                    />
                </label>



                <label>
                    Model Size:
                    <select
                        name="model_size"
                        value={config.model_size}
                        onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange(e)} // Adjust event type here
                        >
                        <option value="small">Small</option>
                        <option value="medium">Medium</option>
                        <option value="large">Large</option>
                        <option value="xlarge">X-Large</option>
                        <option value="xxlarge">XX-Large</option>
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
