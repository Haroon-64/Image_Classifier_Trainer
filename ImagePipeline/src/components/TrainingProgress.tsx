import React, { useEffect, useState } from "react";
import { getTrainingProgress, startTraining } from "../api";
import { Button, CircularProgress } from "@mui/material";

const TrainingProgress: React.FC = () => {
  const [progress, setProgress] = useState({ epoch: 0, total_epochs: 0, status: "Idle" });

  useEffect(() => {
    const interval = setInterval(async () => {
      const response = await getTrainingProgress();
      setProgress(response.data);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async () => {
    await startTraining();
    alert("Training started!");
  };

  return (
    <div>
      <h2>Training Progress</h2>
      <p>Status: {progress.status}</p>
      <p>Epoch: {progress.epoch} / {progress.total_epochs}</p>
      <Button variant="contained" onClick={handleStartTraining}>
        Start Training
      </Button>
      {progress.status === "Training in progress" && <CircularProgress />}
    </div>
  );
};

export default TrainingProgress;