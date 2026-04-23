import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, History as HistoryIcon, Search, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { AppHeader } from "@/components/app-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/use-auth";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/history")({
  component: HistoryPage,
  head: () => ({
    meta: [
      { title: "History — CardioScan" },
      { name: "description", content: "Review your saved cardiomegaly classification results." },
    ],
  }),
});

type Row = {
  id: string;
  image_path: string;
  image_name: string;
  probability: number;
  prediction: number;
  pathology: string;
  patient_name: string | null;
  patient_id: string | null;
  notes: string | null;
  created_at: string;
  signedUrl?: string;
};

function HistoryPage() {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");

  const filteredRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) => {
      const name = r.patient_name?.toLowerCase() ?? "";
      const pid = r.patient_id?.toLowerCase() ?? "";
      const file = r.image_name?.toLowerCase() ?? "";
      return name.includes(q) || pid.includes(q) || file.includes(q);
    });
  }, [rows, query]);

  useEffect(() => {
    if (!authLoading && !user) navigate({ to: "/auth" });
  }, [user, authLoading, navigate]);

  useEffect(() => {
    if (!user) return;
    let active = true;

    const load = async () => {
      setLoading(true);
      const { data, error } = await supabase
        .from("classifications")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(100);

      if (!active) return;
      if (error) {
        toast.error(error.message);
        setLoading(false);
        return;
      }

      const withUrls: Row[] = await Promise.all(
        (data ?? []).map(async (r) => {
          const { data: signed } = await supabase.storage
            .from("xray-uploads")
            .createSignedUrl(r.image_path, 60 * 60);
          return { ...r, signedUrl: signed?.signedUrl };
        }),
      );

      if (!active) return;
      setRows(withUrls);
      setLoading(false);
    };

    load();
    return () => {
      active = false;
    };
  }, [user]);

  const handleDelete = async (row: Row) => {
    const { error: dbErr } = await supabase.from("classifications").delete().eq("id", row.id);
    if (dbErr) {
      toast.error(dbErr.message);
      return;
    }
    await supabase.storage.from("xray-uploads").remove([row.image_path]);
    setRows((prev) => prev.filter((r) => r.id !== row.id));
    toast.success("Removed from history.");
  };

  return (
    <div className="min-h-screen" style={{ background: "var(--gradient-surface)" }}>
      <AppHeader />

      <main className="mx-auto max-w-6xl px-4 pb-20 pt-10 sm:px-6">
        <div className="mb-8 flex items-end justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Classification history</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Your most recent saved scans, newest first.
            </p>
          </div>
          <Button asChild variant="outline" size="sm">
            <Link to="/">New scan</Link>
          </Button>
        </div>

        {!loading && rows.length > 0 && (
          <div className="mb-6 flex items-center gap-3">
            <div className="relative max-w-md flex-1">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by patient name, ID or filename"
                className="pl-9 pr-9"
                aria-label="Search saved scans"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                  aria-label="Clear search"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            <span className="text-xs text-muted-foreground">
              {filteredRows.length} of {rows.length}
            </span>
          </div>
        )}

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="aspect-[4/5] animate-pulse rounded-2xl border border-border bg-muted/40"
              />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <div
            className="flex flex-col items-center rounded-2xl border border-dashed border-border bg-card px-6 py-16 text-center"
            style={{ boxShadow: "var(--shadow-card)" }}
          >
            <HistoryIcon className="mb-3 h-8 w-8 text-muted-foreground" />
            <p className="text-sm font-medium">No saved scans yet</p>
            <p className="mt-1 max-w-sm text-sm text-muted-foreground">
              Run a classification on the home page and save the results to see them here.
            </p>
            <Button asChild className="mt-5" size="sm">
              <Link to="/">Run a scan</Link>
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {rows.map((r) => {
              const positive = r.prediction === 1;
              return (
                <article
                  key={r.id}
                  className="group overflow-hidden rounded-2xl border border-border bg-card"
                  style={{ boxShadow: "var(--shadow-card)" }}
                >
                  <div className="relative aspect-square bg-black">
                    {r.signedUrl ? (
                      <img
                        src={r.signedUrl}
                        alt={r.image_name}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                        Image unavailable
                      </div>
                    )}
                    <button
                      onClick={() => handleDelete(r)}
                      className="absolute right-2 top-2 rounded-full bg-background/80 p-1.5 text-muted-foreground opacity-0 backdrop-blur transition-all hover:bg-background hover:text-destructive group-hover:opacity-100"
                      aria-label="Delete record"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <div className="space-y-2 p-4">
                    {(r.patient_name || r.patient_id) && (
                      <div className="space-y-0.5">
                        {r.patient_name && (
                          <p className="truncate text-sm font-semibold" title={r.patient_name}>
                            {r.patient_name}
                          </p>
                        )}
                        {r.patient_id && (
                          <p className="font-mono text-xs text-muted-foreground">
                            ID: {r.patient_id}
                          </p>
                        )}
                      </div>
                    )}
                    <p className="truncate text-xs text-muted-foreground" title={r.image_name}>
                      {r.image_name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(r.created_at).toLocaleString()}
                    </p>
                    <div className="flex items-center justify-between pt-1">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold",
                        )}
                        style={{
                          backgroundColor: positive
                            ? "color-mix(in oklab, var(--warning) 15%, transparent)"
                            : "color-mix(in oklab, var(--success) 15%, transparent)",
                          color: positive ? "var(--warning)" : "var(--success)",
                        }}
                      >
                        {positive ? (
                          <AlertTriangle className="h-3 w-3" />
                        ) : (
                          <CheckCircle2 className="h-3 w-3" />
                        )}
                        {positive ? "Detected" : "Negative"}
                      </span>
                      <span
                        className="font-mono text-sm tabular-nums"
                        style={{ color: positive ? "var(--warning)" : "var(--success)" }}
                      >
                        {r.probability.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
