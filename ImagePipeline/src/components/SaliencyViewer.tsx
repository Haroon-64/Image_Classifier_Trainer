


import React, { useState } from "react";
import { generateSaliency } from "../api";
import { Button, TextField } from "@mui/material";

const SaliencyViewer: React.FC = () => {
  const [imagePath, setImagePath] = useState("");
  const [saliencyPath, setSaliencyPath] = useState("");

  const handleGenerateSaliency = async () => {
    const response = await generateSaliency(imagePath, 0);
    setSaliencyPath(response.data.path);
  };

  return (
    <div>
      <h2>Saliency Map</h2>
      <TextField
        label="Image Path"
        value={imagePath}
        onChange={(e) => setImagePath(e.target.value)}
        fullWidth
      />
      <Button variant="contained" onClick={handleGenerateSaliency}>Generate Saliency</Button>
      {saliencyPath && <img src={saliencyPath} alt="Saliency Map" />}
    </div>
  );
};

export default SaliencyViewer;


