import axios from "axios";
import type { PredictApiResponse } from "@/types/prediction";

const PREDICT_BASE_URL = import.meta.env.VITE_PREDICT_API_URL ?? "";

const predictClient = axios.create({
  baseURL: PREDICT_BASE_URL,
  timeout: 30000,
});

export async function requestPrediction(file: File): Promise<PredictApiResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const { data } = await predictClient.post<PredictApiResponse>("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return data;
}
