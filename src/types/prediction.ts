export type PredictApiResponse = {
  prediction: string;
  prediction_binary?: 0 | 1;
  mode?: "accurate" | "fast";
  confidence: number; // [0, 1]
  heatmap_url?: string | null;
  source?: "model";
  threshold?: number;
  ensemble_size?: number;
  use_tta?: boolean;
  /** Checkpoint filenames loaded on the inference server (e.g. model_seed8.pth). */
  checkpoints?: string[];
};
