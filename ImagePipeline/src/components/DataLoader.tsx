import React from "react";
import { loadData } from "../api";
import { Button } from "@mui/material";

const DataLoader: React.FC = () => {
  const handleLoadData = async () => {
    try {
      const response = await loadData();
      alert(`Data Loaded: ${response.data.num_samples} samples`);
    } catch (error) {
      alert("Failed to load data.");
    }
  };

  return (
    <div>
      <h2>Data Loader</h2>
      <Button variant="contained" onClick={handleLoadData}>Load Data</Button>
    </div>
  );
};

export default DataLoader;