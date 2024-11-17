import React, { useState } from "react";
import { updateConfig } from "../services/api";

const ConfigForm: React.FC = () => {
  const [dataPath, setDataPath] = useState<string>("");
  const [modelSize, setModelSize] = useState<string>("small");

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const config = {
        data_path: dataPath,
        model_size: modelSize,
        image_size: 224,
        num_classes: 10,
        epochs: 10,
        batch_size: 32,
        learning_rate: 0.001,
        output_path: "./output",
      };
      const response = await updateConfig(config);
      alert(response.message);
    } catch (err) {
      alert("Failed to update config");
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>Data Path:</label>
        <input
          type="text"
          value={dataPath}
          onChange={(e) => setDataPath(e.target.value)}
        />
      </div>
      <div>
        <label>Model Size:</label>
        <select
          value={modelSize}
          onChange={(e) => setModelSize(e.target.value)}
        >
          <option value="small">Small</option>
          <option value="medium">Medium</option>
          <option value="large">Large</option>
        </select>
      </div>
      <button type="submit">Update Config</button>
    </form>
  );
};

export default ConfigForm;
