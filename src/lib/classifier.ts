import { isAxiosError } from "axios";
import { requestPrediction } from "@/services/predict";

export type PredictionSource = "model" | "mock";

export type ClassificationResult = {
  label: string;
  probability: number; // [0, 1]
  prediction: 0 | 1;
  heatmapUrl?: string;
  /** Where this prediction came from. Always set so the UI can prove it. */
  source: PredictionSource;
};

export type InferenceMode = "accurate" | "fast";

const DECISION_THRESHOLD = 0.5;

/**
 * Hash-based deterministic mock used ONLY when explicitly requested via the
 * `VITE_USE_MOCK_PREDICTION=true` env var. Never used as a silent fallback.
 */
function pseudoHash(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 0xffffffff;
}

async function mockClassify(file: File): Promise<ClassificationResult> {
  const latency = 700 + Math.random() * 800;
  await new Promise((r) => setTimeout(r, latency));
  const seed = pseudoHash(`${file.name}:${file.size}`);
  const probability = Math.min(0.99, Math.max(0.01, seed));
  const prediction: 0 | 1 = probability >= DECISION_THRESHOLD ? 1 : 0;
  const label = prediction === 1 ? PATHOLOGY_LABEL : `No ${PATHOLOGY_LABEL} indication`;
  return { label, probability, prediction, source: "mock" };
}

function explainAxiosError(error: unknown): string {
  if (isAxiosError(error)) {
    if (error.code === "ECONNABORTED") return "Model server timed out.";
    if (error.response) {
      const detail =
        (error.response.data as { detail?: string } | undefined)?.detail ?? error.response.statusText;
      return `Model server returned ${error.response.status}: ${detail}`;
    }
    if (error.request) {
      return `Model server unreachable at ${error.config?.baseURL ?? "(no base URL)"}. Is the inference server running?`;
    }
    return error.message;
  }
  return (error as Error).message ?? "Unknown error";
}

export async function classifyImage(
  file: File,
  options: { mode?: InferenceMode } = {},
): Promise<ClassificationResult> {
  const useMock = import.meta.env.VITE_USE_MOCK_PREDICTION === "true";

  if (useMock) {
    console.warn("[classifier] using MOCK predictions (VITE_USE_MOCK_PREDICTION=true)");
    return mockClassify(file);
  }

  try {
    const mode = options.mode ?? "accurate";
    const response = await requestPrediction(file, {
      mode,
      useTta: mode !== "fast",
      maxModels: mode === "fast" ? 1 : undefined,
    });
    const probability = Math.min(1, Math.max(0, response.confidence));
    // Use numeric server output directly when present.
    const threshold = response.threshold ?? DECISION_THRESHOLD;
    const prediction: 0 | 1 =
      response.prediction_binary === 0 || response.prediction_binary === 1
        ? response.prediction_binary
        : probability >= threshold
          ? 1
          : 0;
    return {
      label: response.prediction,
      probability,
      prediction,
      heatmapUrl: response.heatmap_url ?? undefined,
      source: "model",
    };
  } catch (error) {
    throw new Error(explainAxiosError(error));
  }
}

export const PATHOLOGY_LABEL = "Cardiomegaly";
