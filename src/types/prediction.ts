export type PredictApiResponse = {
  prediction: string;
  confidence: number; // [0, 1]
  heatmap_url?: string | null;
};
