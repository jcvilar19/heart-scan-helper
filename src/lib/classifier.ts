/**
 * Mocked cardiomegaly classifier.
 *
 * Replace `mockClassify` with a real `fetch()` call to your Python service
 * (see /model/predict.py) once it is deployed. The contract is intentionally
 * minimal so the UI doesn't need to change.
 */

export type ClassificationResult = {
  probability: number; // [0, 1]
  prediction: 0 | 1;
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

export async function classifyImage(file: File): Promise<ClassificationResult> {
  // Simulate network + inference latency (700-1500ms)
  const latency = 700 + Math.random() * 800;
  await new Promise((r) => setTimeout(r, latency));

  const seed = pseudoHash(`${file.name}:${file.size}`);
  // Bias slightly toward the middle of the range for a realistic spread
  const probability = Math.min(0.99, Math.max(0.01, seed));
  const prediction: 0 | 1 = probability >= DECISION_THRESHOLD ? 1 : 0;

  return { probability, prediction };
}

export const PATHOLOGY_LABEL = "Cardiomegaly";
