import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000", // Backend URL
});

export const getConfig = async () => {
  const response = await API.get("/config");
  return response.data;
};

export const updateConfig = async (config: object) => {
  const response = await API.post("/config", config);
  return response.data;
};

// Add other API functions as needed
export const loadData = async () => API.post("/data");
export const buildModel = async () => API.post("/model");
export const trainModel = async () => API.post("/train");
export const getStatus = async () => API.get("/status");
