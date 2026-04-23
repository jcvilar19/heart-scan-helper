import { env, pipeline, type ZeroShotImageClassificationPipeline } from "@huggingface/transformers";

// We only want to load models from the HuggingFace Hub, never from the app's
// own `/models/...` path (which would 404 on Cloudflare Pages).
env.allowLocalModels = false;
// Do NOT opt into the browser Cache Storage API: transformers.js throws
// "Browser cache is not available in this environment." in contexts where
// `caches` isn't accessible (CF Workers, some preview sandboxes, certain
// privacy modes). The browser's HTTP cache still caches HF CDN responses.
env.useBrowserCache = false;

const CLIP_MODEL_ID = "Xenova/clip-vit-base-patch32";

const CHEST_XRAY_LABEL = "a chest x-ray";

// We phrase each candidate as a full sentence; CLIP's zero-shot template wraps
// them in "This is a photo of {}", so keeping them noun-phrase-like works best.
const CANDIDATE_LABELS = [
  CHEST_XRAY_LABEL,
  "a medical scan of another body part",
  "a photograph of a person",
  "a photograph of an animal",
  "a photograph of an object or scene",
  "a screenshot or document",
  "a drawing or illustration",
];

// Minimum CLIP score required for the top label to be trusted as a chest X-ray.
// CLIP softmaxes across candidates, so a clear chest X-ray typically scores
// > 0.8 on this label set while ambiguous inputs stay below 0.5.
const CLIP_SCORE_THRESHOLD = 0.5;

// --- Heuristic thresholds -------------------------------------------------

const MIN_DIMENSION = 256;
// Average per-pixel chroma (max(R,G,B) - min(R,G,B)) above which we consider
// the image a color photograph. Real X-rays are essentially grayscale, so the
// average chroma is close to 0. We leave headroom for JPEG color drift.
const MAX_COLOR_CHROMA = 18;
const SAMPLE_SIZE = 128;

// --- Public types ---------------------------------------------------------

export type ImageValidationResult =
  | { ok: true; topLabel: string; score: number }
  | {
      ok: false;
      stage: "heuristic" | "clip" | "load-error";
      /** Short, user-facing message. Safe to show directly in toasts / cards. */
      reason: string;
      /** Longer diagnostic detail, only for logs. */
      detail?: string;
    };

export type ValidatorProgress =
  | { status: "loading" }
  | { status: "downloading"; file: string; progress: number }
  | { status: "ready" }
  | { status: "running" };

type ProgressHandler = (p: ValidatorProgress) => void;

// --- CLIP pipeline (lazy, cached) ----------------------------------------

let classifierPromise: Promise<ZeroShotImageClassificationPipeline> | null = null;

function getClassifier(onProgress?: ProgressHandler): Promise<ZeroShotImageClassificationPipeline> {
  if (classifierPromise) return classifierPromise;

  classifierPromise = pipeline("zero-shot-image-classification", CLIP_MODEL_ID, {
    progress_callback: (data: unknown) => {
      if (!onProgress) return;
      const evt = data as { status?: string; file?: string; progress?: number };
      if (evt.status === "progress") {
        onProgress({
          status: "downloading",
          file: evt.file ?? "model",
          progress: typeof evt.progress === "number" ? evt.progress : 0,
        });
      } else if (evt.status === "ready") {
        onProgress({ status: "ready" });
      } else if (evt.status === "initiate" || evt.status === "download") {
        onProgress({ status: "loading" });
      }
    },
  }) as Promise<ZeroShotImageClassificationPipeline>;

  // If loading fails, reset so the next call can retry.
  classifierPromise.catch(() => {
    classifierPromise = null;
  });

  return classifierPromise;
}

/** Preload the CLIP model in the background. Safe to call repeatedly. */
export function warmupValidator(onProgress?: ProgressHandler): Promise<unknown> {
  return getClassifier(onProgress);
}

// --- Heuristic stage ------------------------------------------------------

async function heuristicCheck(file: File): Promise<{ ok: true } | { ok: false; reason: string }> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return { ok: false, reason: "Could not decode image." };
  }

  const { width, height } = bitmap;
  if (width < MIN_DIMENSION || height < MIN_DIMENSION) {
    bitmap.close();
    return {
      ok: false,
      reason: `Image is too small (${width}×${height}). This is not a valid chest X-ray image.`,
    };
  }

  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE_SIZE;
  canvas.height = SAMPLE_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    bitmap.close();
    return { ok: false, reason: "Could not analyze image pixels." };
  }
  ctx.drawImage(bitmap, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  const { data } = ctx.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  bitmap.close();

  const pixelCount = data.length / 4;
  let totalChroma = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    totalChroma += max - min;
  }
  const avgChroma = totalChroma / pixelCount;

  if (avgChroma > MAX_COLOR_CHROMA) {
    return {
      ok: false,
      reason: "This is not a valid chest X-ray image (color photograph detected).",
    };
  }

  return { ok: true };
}

// --- CLIP stage -----------------------------------------------------------

async function clipCheck(file: File, onProgress?: ProgressHandler): Promise<ImageValidationResult> {
  let classifier: ZeroShotImageClassificationPipeline;
  try {
    classifier = await getClassifier(onProgress);
  } catch (err) {
    return {
      ok: false,
      stage: "load-error",
      reason: "Could not verify the image. Please try again in a moment.",
      detail: (err as Error).message,
    };
  }

  const url = URL.createObjectURL(file);
  try {
    onProgress?.({ status: "running" });
    const raw = await classifier(url, CANDIDATE_LABELS);

    const output = (Array.isArray(raw[0]) ? raw[0] : raw) as Array<{
      label: string;
      score: number;
    }>;
    const sorted = [...output].sort((a, b) => b.score - a.score);
    const top = sorted[0];

    if (top?.label === CHEST_XRAY_LABEL && top.score >= CLIP_SCORE_THRESHOLD) {
      return { ok: true, topLabel: top.label, score: top.score };
    }

    const friendlyLabel = (top?.label ?? "unknown").replace(/^a /i, "");
    const pct = top ? `${Math.round(top.score * 100)}%` : "low";
    return {
      ok: false,
      stage: "clip",
      reason: "This is not a valid chest X-ray image.",
      detail: `Top CLIP label: "${friendlyLabel}" (${pct}).`,
    };
  } catch (err) {
    return {
      ok: false,
      stage: "load-error",
      reason: "Could not verify the image. Please try again in a moment.",
      detail: (err as Error).message,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

// --- Combined entry point -------------------------------------------------

export async function validateChestXray(
  file: File,
  onProgress?: ProgressHandler,
): Promise<ImageValidationResult> {
  const heuristic = await heuristicCheck(file);
  if (!heuristic.ok) {
    return { ok: false, stage: "heuristic", reason: heuristic.reason };
  }
  return clipCheck(file, onProgress);
}

/** Human-readable message to display in toasts / cards for any rejection. */
export function describeRejection(result: ImageValidationResult): string {
  if (result.ok) return "";
  return result.reason;
}
