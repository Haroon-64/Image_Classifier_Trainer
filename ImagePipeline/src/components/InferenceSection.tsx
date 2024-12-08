import React, { useState } from "react";
import { performInference } from "../api";
import { Button, TextField } from "@mui/material";

const InferenceSection: React.FC = () => {
  const [imagePath, setImagePath] = useState("");
  const [result, setResult] = useState("");

  const handleInference = async () => {
    const response = await performInference(imagePath);
    setResult(`Predicted Class: ${response.data.predicted_class}`);
  };

  return (
    <div>
      <h2>Inference</h2>
      <TextField
        label="Image Path"
        value={imagePath}
        onChange={(e) => setImagePath(e.target.value)}
        fullWidth
      />
      <Button variant="contained" onClick={handleInference}>Run Inference</Button>
      {result && <p>{result}</p>}
    </div>
  );
};

export default InferenceSection;