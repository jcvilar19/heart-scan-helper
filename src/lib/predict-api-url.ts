/**
 * Resolves the inference API base URL for the current browser context.
 *
 * If the app is opened via Vite's "Network" URL (e.g. `http://192.168.1.5:8080`)
 * but `VITE_PREDICT_API_URL` is `http://127.0.0.1:8000`, a direct request to
 * 127.0.0.1 from another device is wrong (it targets that device). In that
 * case we use the same hostname as the page so the API hits the machine
 * running Vite and uvicorn.
 *
 * Unchanged: HTTPS production (Lovable, ngrok), and env URLs that already
 * use a non-loopback host.
 */
export function resolvePredictApiBaseUrl(): string {
  const configured = import.meta.env.VITE_PREDICT_API_URL ?? "";
  if (!configured) {
    throw new Error(
      "VITE_PREDICT_API_URL is not configured. Add it to .env and restart the dev server.",
    );
  }
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
