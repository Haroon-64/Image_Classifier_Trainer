import React from "react";
import { buildModel, saveModel } from "../api";
import { Button } from "@mui/material";

const ModelBuilder: React.FC = () => {
  const handleBuildModel = async () => {
    try {
      await buildModel();
      alert("Model built successfully!");
    } catch (error) {
      alert("Failed to build model.");
    }
  };

  const handleSaveModel = async () => {
    const path = prompt("Enter model save path:");
    if (path) {
      await saveModel(path);
      alert("Model saved!");
    }
  };

  return (
    <div>
      <h2>Model Builder</h2>
      <Button variant="contained" onClick={handleBuildModel}>Build Model</Button>
      <Button variant="contained" onClick={handleSaveModel}>Save Model</Button>
    </div>
  );
};

export default ModelBuilder;