import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import axios from "axios";
import { XCircle } from "lucide-react";
import { AppHeader } from "@/components/app-header";
import { ImageDropzone } from "@/components/image-dropzone";
import { ResultsGallery, type AnalysisItem } from "@/components/result-card";
import { Button } from "@/components/ui/button";
import { classifyImage } from "@/lib/classifier";
import { resolvePredictApiBaseUrl } from "@/lib/predict-api-url";
import { PREDICT_DEFAULT_HEADERS } from "@/services/predict";
import { validateChestXray, warmupValidator } from "@/lib/image-validator";
import { useAuth } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";

type HealthState =
  | { status: "unknown" }
  | { status: "checking" }
  | { status: "ok"; checkpoints: string[]; threshold: number; useTta: boolean }
  | { status: "mock" }
  | { status: "down"; message: string };

export const Route = createFileRoute("/app")({
  component: AppPage,
  head: () => ({
    meta: [
      { title: "Scanner — CardioScan" },
      {
        name: "description",
        content: "Upload chest X-rays and get an AI cardiomegaly probability score.",
      },
    ],
  }),
});

function AppPage() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [items, setItems] = useState<AnalysisItem[]>([]);
  const [health, setHealth] = useState<HealthState>({ status: "unknown" });
  // Tracks which item ids we've already kicked off `saveItem` for, so React
  // strict-mode double-renders don't trigger duplicate uploads.
  const saveAttempted = useRef<Set<string>>(new Set());
  // Per-item debounce timers for "update existing classification row"
  // (patient name / patient id / notes edits after the initial save).
  const updateTimers = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    if (!loading && !user) {
      navigate({ to: "/auth" });
    }
  }, [loading, user, navigate]);

  useEffect(() => {
    if (!user) return;
    // Start downloading the CLIP model in the background so the first upload
    // does not wait on ~150 MB of weights. The browser caches the weights for
    // later sessions.
    void warmupValidator();
  }, [user]);

  const checkHealth = useCallback(async () => {
    const useMock = import.meta.env.VITE_USE_MOCK_PREDICTION === "true";
    if (useMock) {
      setHealth({ status: "mock" });
      return;
    }
    if (!import.meta.env.VITE_PREDICT_API_URL) {
      setHealth({ status: "down", message: "VITE_PREDICT_API_URL is not set in .env" });
      return;
    }
    const apiUrl = resolvePredictApiBaseUrl();
    setHealth({ status: "checking" });
    try {
      const { data } = await axios.get(`${apiUrl}/health`, {
        timeout: 8000,
        headers: PREDICT_DEFAULT_HEADERS,
      });
      setHealth({
        status: "ok",
        checkpoints: (data?.checkpoints as string[] | undefined) ?? [],
        threshold: typeof data?.threshold === "number" ? data.threshold : 0.5,
        useTta: Boolean(data?.use_tta),
      });
    } catch (err) {
      let msg = axios.isAxiosError(err)
        ? err.code === "ECONNABORTED"
          ? "Health check timed out"
          : err.message
        : (err as Error).message;

      if (axios.isAxiosError(err) && err.message === "Network Error" && typeof window !== "undefined") {
        const page = window.location;
        if (page.protocol === "https:" && apiUrl.startsWith("http:")) {
          msg =
            "This page is HTTPS (e.g. Lovable) but the API URL is HTTP — the browser blocks that (mixed content). Set VITE_PREDICT_API_URL to an https URL (e.g. your ngrok URL).";
        } else {
          msg = `${msg} Is uvicorn running? From inference_server: source .venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port 8000`;
        }
      }
      setHealth({ status: "down", message: msg });
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    void checkHealth();
  }, [user, checkHealth]);

  // Auto-retry every 10s while the server is unreachable, so the banner
  // turns green on its own once the user starts the inference server.
  useEffect(() => {
    if (health.status !== "down") return;
    const t = setInterval(() => {
      void checkHealth();
    }, 10000);
    return () => clearInterval(t);
  }, [health.status, checkHealth]);

  const handleFiles = async (files: File[]) => {
    const newItems: AnalysisItem[] = files.map((file) => ({
      id: `${file.name}-${file.size}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      file,
      previewUrl: URL.createObjectURL(file),
      status: "pending",
      notes: "",
      patientName: "",
      patientId: "",
      saveStatus: "idle",
    }));

    setItems((prev) => [...newItems, ...prev]);

    for (const item of newItems) {
      setItems((prev) => prev.map((i) => (i.id === item.id ? { ...i, status: "validating" } : i)));
      try {
        const validation = await validateChestXray(item.file);
        if (!validation.ok) {
          if (validation.detail) {
            console.warn("[validator]", item.file.name, validation.detail);
          }
          setItems((prev) => {
            const target = prev.find((i) => i.id === item.id);
            if (target) URL.revokeObjectURL(target.previewUrl);
            return prev.filter((i) => i.id !== item.id);
          });
          if (validation.stage === "load-error") {
            toast.error(`${item.file.name}: ${validation.reason}`);
          } else {
            toast.error(`${item.file.name} is not a valid chest X-ray image.`);
          }
          continue;
        }

        setItems((prev) => prev.map((i) => (i.id === item.id ? { ...i, status: "analyzing" } : i)));
        const result = await classifyImage(item.file);
        setItems((prev) =>
          prev.map((i) => (i.id === item.id ? { ...i, status: "done", result } : i)),
        );
      } catch (err) {
        const message = (err as Error).message ?? "Prediction failed.";
        toast.error(`${item.file.name}: ${message}`);
        setItems((prev) =>
          prev.map((i) => (i.id === item.id ? { ...i, status: "error", error: message } : i)),
        );
      }
    }
  };

  const handleRemove = (id: string) => {
    setItems((prev) => {
      const target = prev.find((i) => i.id === id);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((i) => i.id !== id);
    });
    const t = updateTimers.current.get(id);
    if (t) {
      window.clearTimeout(t);
      updateTimers.current.delete(id);
    }
  };

  const handleClear = () => {
    items.forEach((i) => URL.revokeObjectURL(i.previewUrl));
    setItems([]);
    updateTimers.current.forEach((t) => window.clearTimeout(t));
    updateTimers.current.clear();
  };

  // Update an already-saved row in Supabase with the latest patient info / notes.
  // Debounced per-item so typing doesn't fire a request per keystroke.
  const scheduleRemoteUpdate = useCallback(
    (id: string) => {
      if (!user) return;
      const existing = updateTimers.current.get(id);
      if (existing) window.clearTimeout(existing);
      const handle = window.setTimeout(async () => {
        updateTimers.current.delete(id);
        // Read the latest item state via a no-op setItems call so we don't
        // capture stale values from the debounce closure.
        const item = await new Promise<AnalysisItem | undefined>((resolve) => {
          setItems((prev) => {
            resolve(prev.find((i) => i.id === id));
            return prev;
          });
        });
        if (!item || !item.savedRecordId) return;
        const { error } = await supabase
          .from("classifications")
          .update({
            patient_name: item.patientName.trim() || null,
            patient_id: item.patientId.trim() || null,
            notes: item.notes.trim() || null,
          })
          .eq("id", item.savedRecordId);
        if (error) {
          console.warn("[history] update failed", error);
        }
      }, 600);
      updateTimers.current.set(id, handle);
    },
    [user],
  );

  const handleNotesChange = (id: string, notes: string) => {
    setItems((prev) => prev.map((i) => (i.id === id ? { ...i, notes } : i)));
    scheduleRemoteUpdate(id);
  };

  const handlePatientChange = (
    id: string,
    fields: { patientName?: string; patientId?: string },
  ) => {
    setItems((prev) =>
      prev.map((i) =>
        i.id === id
          ? {
              ...i,
              patientName: fields.patientName ?? i.patientName,
              patientId: fields.patientId ?? i.patientId,
            }
          : i,
      ),
    );
    scheduleRemoteUpdate(id);
  };

  // Persist a single freshly-finished analysis to Supabase storage + DB.
  // Called automatically by the effect below as soon as an item turns "done".
  const saveItem = useCallback(
    async (item: AnalysisItem) => {
      if (!user || !item.result) return;
      setItems((prev) =>
        prev.map((i) =>
          i.id === item.id ? { ...i, saveStatus: "saving", saveError: undefined } : i,
        ),
      );
      try {
        const ext = item.file.name.split(".").pop() ?? "jpg";
        const path = `${user.id}/${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${ext}`;
        const { error: upErr } = await supabase.storage
          .from("xray-uploads")
          .upload(path, item.file, { contentType: item.file.type, upsert: false });
        if (upErr) throw upErr;

        const { data: inserted, error: insErr } = await supabase
          .from("classifications")
          .insert({
            user_id: user.id,
            image_path: path,
            image_name: item.file.name,
            probability: item.result.probability,
            prediction: item.result.prediction,
            pathology: "cardiomegaly",
            patient_name: item.patientName.trim() || null,
            patient_id: item.patientId.trim() || null,
            notes: item.notes.trim() || null,
          })
          .select("id")
          .single();
        if (insErr) throw insErr;

        setItems((prev) =>
          prev.map((i) =>
            i.id === item.id
              ? { ...i, saveStatus: "saved", savedRecordId: inserted?.id }
              : i,
          ),
        );
      } catch (err) {
        const message = (err as Error).message ?? "Failed to save to history.";
        console.warn("[history] save failed", err);
        setItems((prev) =>
          prev.map((i) =>
            i.id === item.id ? { ...i, saveStatus: "error", saveError: message } : i,
          ),
        );
        toast.error(`History: ${message}`);
      }
    },
    [user],
  );

  // Auto-save any freshly-completed analysis exactly once.
  useEffect(() => {
    if (!user) return;
    const candidates = items.filter(
      (i) =>
        i.status === "done" &&
        i.result &&
        i.saveStatus === "idle" &&
        !saveAttempted.current.has(i.id),
    );
    for (const c of candidates) {
      saveAttempted.current.add(c.id);
      void saveItem(c);
    }
  }, [items, user, saveItem]);

  if (loading || !user) {
    return (
      <div className="min-h-screen" style={{ background: "var(--gradient-surface)" }}>
        <AppHeader />
        <div className="mx-auto max-w-3xl px-4 py-24 text-center text-sm text-muted-foreground">
          Checking your session…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{ background: "var(--gradient-surface)" }}>
      <AppHeader />

      <main className="mx-auto max-w-6xl px-4 pb-20 pt-10 sm:px-6">
        <section className="mx-auto max-w-3xl text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
            Signed in · Ready to analyze
          </div>
          <h1 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            Upload chest X-rays for{" "}
            <span
              className="bg-clip-text text-transparent"
              style={{ backgroundImage: "var(--gradient-primary)" }}
            >
              cardiomegaly screening
            </span>
          </h1>
          <p className="mx-auto mt-3 max-w-2xl text-pretty text-sm text-muted-foreground sm:text-base">
            Drop one or more images. Each receives a probability score and a binary verdict in
            seconds.
          </p>
        </section>

        <section className="mx-auto mt-6 max-w-3xl">
          <HealthBanner health={health} onRetry={checkHealth} />
        </section>

        <section className="mx-auto mt-4 max-w-3xl">
          <ImageDropzone onFiles={handleFiles} />
        </section>

        <div className="mx-auto mt-10 max-w-6xl">
          <ResultsGallery
            items={items}
            onRemove={handleRemove}
            onClear={handleClear}
            onNotesChange={handleNotesChange}
            onPatientChange={handlePatientChange}
          />
        </div>

        <p className="mx-auto mt-16 max-w-2xl text-center text-xs text-muted-foreground">
          For research and educational use only. Not a substitute for professional medical
          diagnosis. <Link to="/" className="underline hover:text-foreground">Back to home</Link>
        </p>
      </main>
    </div>
  );
}

function HealthBanner({
  health,
  onRetry,
}: {
  health: HealthState;
  onRetry: () => void | Promise<void>;
}) {
  if (health.status !== "down") {
    return null;
  }
  return (
    <div className="flex items-start justify-between gap-3 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-800">
      <div className="flex items-start gap-2">
        <XCircle className="mt-0.5 h-3.5 w-3.5" />
        <div className="space-y-1">
          <div className="font-medium">Service temporarily unavailable.</div>
          <div className="text-red-700/90">
            We can’t reach the analysis service right now. Please try again in a moment.
          </div>
        </div>
      </div>
      <Button size="sm" variant="outline" onClick={() => void onRetry()}>
        Retry
      </Button>
    </div>
  );
}
