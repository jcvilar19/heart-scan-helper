/**
 * Resolves the inference API base URL for the current browser context.
 *
 * Defaults:
 *   - **Vite dev** (`npm run dev`): `http://127.0.0.1:8000` so predictions use
 *     your local `uvicorn` checkpoints — not a remote Hugging Face Space.
 *   - **Production build** without `VITE_PREDICT_API_URL`: demo HF Space (below).
 *
 * If the app is opened via Vite's "Network" URL (e.g. `http://192.168.1.5:8080`)
 * but the resolved URL still points at `127.0.0.1`, requests from another device
 * would hit the wrong host. For private LAN page hosts we rewrite loopback API
 * URLs to the page hostname so the API reaches the machine running uvicorn.
 */
/** Demo inference API when no env is set in a production bundle. */
const DEFAULT_PREDICT_API_URL = "https://jcvilar-cardio-scan-api.hf.space";

const DEV_LOCAL_PREDICT_API_URL = "http://127.0.0.1:8000";

export function resolvePredictApiBaseUrl(): string {
  const configured =
    import.meta.env.VITE_PREDICT_API_URL ??
    (import.meta.env.DEV ? DEV_LOCAL_PREDICT_API_URL : DEFAULT_PREDICT_API_URL);
  if (typeof window === "undefined") {
    return configured;
  }
  const pageHost = window.location.hostname;
  const pageIsPrivateLan =
    /^192\.168\./.test(pageHost) || /^10\./.test(pageHost) || /^172\.(1[6-9]|2\d|3[01])\./.test(pageHost);
  if (!pageIsPrivateLan) {
    return configured;
  }
  try {
    const u = new URL(configured);
    if (u.hostname === "127.0.0.1" || u.hostname === "localhost") {
      u.hostname = pageHost;
      return u.origin;
    }
  } catch {
    // leave configured as-is
  }
  return configured;
}
