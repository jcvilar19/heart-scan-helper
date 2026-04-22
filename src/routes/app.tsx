import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Save } from "lucide-react";
import { AppHeader } from "@/components/app-header";
import { ImageDropzone } from "@/components/image-dropzone";
import { ResultsGallery, type AnalysisItem } from "@/components/result-card";
import { Button } from "@/components/ui/button";
import { classifyImage } from "@/lib/classifier";
import { useAuth } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";

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
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      navigate({ to: "/auth" });
    }
  }, [loading, user, navigate]);

  const allDone = useMemo(
    () => items.length > 0 && items.every((i) => i.status === "done" || i.status === "error"),
    [items],
  );
  const successful = useMemo(() => items.filter((i) => i.status === "done"), [items]);

  const handleFiles = async (files: File[]) => {
    const newItems: AnalysisItem[] = files.map((file) => ({
      id: `${file.name}-${file.size}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      file,
      previewUrl: URL.createObjectURL(file),
      status: "pending",
      notes: "",
    }));

    setItems((prev) => [...newItems, ...prev]);

    for (const item of newItems) {
      setItems((prev) => prev.map((i) => (i.id === item.id ? { ...i, status: "analyzing" } : i)));
      try {
        const result = await classifyImage(item.file);
        setItems((prev) =>
          prev.map((i) => (i.id === item.id ? { ...i, status: "done", result } : i)),
        );
      } catch (err) {
        setItems((prev) =>
          prev.map((i) =>
            i.id === item.id ? { ...i, status: "error", error: (err as Error).message } : i,
          ),
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
  };

  const handleClear = () => {
    items.forEach((i) => URL.revokeObjectURL(i.previewUrl));
    setItems([]);
  };

  const handleNotesChange = (id: string, notes: string) => {
    setItems((prev) => prev.map((i) => (i.id === id ? { ...i, notes } : i)));
  };

  const handleSave = async () => {
    if (!user) {
      toast.error("Please sign in to save results.");
      return;
    }
    if (successful.length === 0) {
      toast.info("Nothing to save yet.");
      return;
    }
    setSaving(true);
    try {
      let saved = 0;
      for (const item of successful) {
        if (!item.result) continue;
        const ext = item.file.name.split(".").pop() ?? "jpg";
        const path = `${user.id}/${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${ext}`;
        const { error: upErr } = await supabase.storage
          .from("xray-uploads")
          .upload(path, item.file, { contentType: item.file.type, upsert: false });
        if (upErr) throw upErr;

        const { error: insErr } = await supabase.from("classifications").insert({
          user_id: user.id,
          image_path: path,
          image_name: item.file.name,
          probability: item.result.probability,
          prediction: item.result.prediction,
          pathology: "cardiomegaly",
        });
        if (insErr) throw insErr;
        saved += 1;
      }
      toast.success(`Saved ${saved} result${saved === 1 ? "" : "s"} to your history.`);
    } catch (err) {
      toast.error((err as Error).message ?? "Failed to save results.");
    } finally {
      setSaving(false);
    }
  };

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

        <section className="mx-auto mt-8 max-w-3xl">
          <ImageDropzone onFiles={handleFiles} />
          {items.length > 0 && (
            <div className="mt-3 flex items-center justify-end gap-2">
              {allDone && successful.length > 0 && (
                <Button size="sm" onClick={handleSave} disabled={saving}>
                  <Save className="h-4 w-4" />
                  {saving ? "Saving…" : `Save ${successful.length} to history`}
                </Button>
              )}
            </div>
          )}
        </section>

        <div className="mx-auto mt-10 max-w-6xl">
          <ResultsGallery
            items={items}
            onRemove={handleRemove}
            onClear={handleClear}
            onNotesChange={handleNotesChange}
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
