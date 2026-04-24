import axios from "axios";
import { resolvePredictApiBaseUrl } from "@/lib/predict-api-url";
import type { PredictApiResponse } from "@/types/prediction";

export type PredictionRequestOptions = {
  mode?: "accurate" | "fast";
  useTta?: boolean;
  maxModels?: number;
};

/**
 * Headers added to every call to the inference server.
 *
 * `ngrok-skip-browser-warning` tells ngrok's free-tier edge to bypass the
 * HTML interstitial it would otherwise inject on the first request from a
 * new browser. Harmless against any non-ngrok backend (the server simply
 * ignores the header).
 */
export const PREDICT_DEFAULT_HEADERS: Record<string, string> = {
  "ngrok-skip-browser-warning": "true",
};

function getBaseUrl(): string {
  return resolvePredictApiBaseUrl();
}

export async function requestPrediction(
  file: File,
  options: PredictionRequestOptions = {},
): Promise<PredictApiResponse> {
  const baseURL = getBaseUrl();
  const formData = new FormData();
  formData.append("image", file);

  if (import.meta.env.DEV && baseURL !== import.meta.env.VITE_PREDICT_API_URL) {
    console.info(`[predict] API base resolved for LAN: ${baseURL} (from VITE_PREDICT_API_URL)`);
  }
  console.info(`[predict] POST ${baseURL}/predict  file=${file.name}  size=${file.size}`);
  const { data } = await axios.post<PredictApiResponse>(`${baseURL}/predict`, formData, {
    headers: { ...PREDICT_DEFAULT_HEADERS, "Content-Type": "multipart/form-data" },
    timeout: 60000,
    params:
      options.mode === undefined && options.useTta === undefined && options.maxModels === undefined
        ? undefined
        : {
            ...(options.mode === undefined ? {} : { mode: options.mode }),
            ...(options.useTta === undefined ? {} : { use_tta: options.useTta }),
            ...(options.maxModels === undefined ? {} : { max_models: options.maxModels }),
          },
  });
  console.info("[predict] response:", data);
  if (data.checkpoints?.length) {
    console.log("[predict] ensemble checkpoints (.pth):", data.checkpoints);
  }
  return data;
}
