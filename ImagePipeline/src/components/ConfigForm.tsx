import React, { useState } from "react";
import { updateConfig } from "../api";
import { TextField, Button } from "@mui/material";

const ConfigForm: React.FC = () => {
  const [config, setConfig] = useState({
    data_path: "",
    model_name: "resnet",
    model_size: "small",
    image_size: 224,
    num_classes: 10,
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    output_path: "./output",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setConfig({ ...config, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      const response = await updateConfig(config);
      alert("Configuration updated successfully!");
    } catch (error) {
      alert("Failed to update configuration.");
    }
  };

  return (
    <div>
      <h2>Model Configuration</h2>
      <TextField name="data_path" label="Data Path" onChange={handleChange} fullWidth />
      <TextField name="output_path" label="Output Path" onChange={handleChange} fullWidth />
      <TextField name="num_classes" label="Number of Classes" onChange={handleChange} type="number" fullWidth />
      <Button variant="contained" onClick={handleSubmit}>Save Configuration</Button>
    </div>
  );
};

export default ConfigForm;
