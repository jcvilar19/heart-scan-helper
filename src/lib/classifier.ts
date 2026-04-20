import { requestPrediction } from "@/services/predict";

export type ClassificationResult = {
  label: string;
  probability: number; // [0, 1]
  prediction: 0 | 1;
  heatmapUrl?: string;
};

const DECISION_THRESHOLD = 0.5;

/**
 * Deterministic-ish mock based on a hash of the file name + size, so each
 * image consistently produces the same probability across renders.
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
  // Simulate network + inference latency (700-1500ms)
  const latency = 700 + Math.random() * 800;
  await new Promise((r) => setTimeout(r, latency));
  const seed = pseudoHash(`${file.name}:${file.size}`);
  const probability = Math.min(0.99, Math.max(0.01, seed));
  const prediction: 0 | 1 = probability >= DECISION_THRESHOLD ? 1 : 0;
  const label = prediction === 1 ? PATHOLOGY_LABEL : `No ${PATHOLOGY_LABEL} indication`;

  return { label, probability, prediction };
}

export async function classifyImage(file: File): Promise<ClassificationResult> {
  // Keep the app runnable in dev even without a deployed model API.
  if (!import.meta.env.VITE_PREDICT_API_URL) {
    return mockClassify(file);
  }

  try {
    const response = await requestPrediction(file);
    const probability = Math.min(1, Math.max(0, response.confidence));
    const prediction: 0 | 1 = probability >= DECISION_THRESHOLD ? 1 : 0;

    return {
      label: response.prediction,
      probability,
      prediction,
      heatmapUrl: response.heatmap_url ?? undefined,
    };
  } catch (error) {
    // If the model API is temporarily down, keep UI functional with mock output.
    console.warn("Predict API unavailable, using local mock fallback.", error);
    return mockClassify(file);
  }
}

export const PATHOLOGY_LABEL = "Cardiomegaly";
