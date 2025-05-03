import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

export const updateConfig = (config: any) => axios.post(`${API_BASE_URL}/config`, config);
export const loadData = () => axios.post(`${API_BASE_URL}/data`);
export const buildModel = () => axios.post(`${API_BASE_URL}/model`);
export const saveModel = (path: string) => axios.post(`${API_BASE_URL}/model/save`, { model_path: path });
export const loadModel = (path: string) => axios.post(`${API_BASE_URL}/model/load`, { model_path: path });
export const startTraining = () => axios.post(`${API_BASE_URL}/train`);
export const getTrainingProgress = () => axios.get(`${API_BASE_URL}/train/progress`);
export const generateSaliency = (imagePath: string, targetClass: number) =>
  axios.post(`${API_BASE_URL}/saliency`, { image_path: imagePath, target_class: targetClass });
export const performInference = (imagePath: string) =>
  axios.post(`${API_BASE_URL}/inference`, { image_path: imagePath });
