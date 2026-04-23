import { useState } from "react";
import type { ComponentProps } from "react";
import { Loader2, Sparkles, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";

export type ScanForInsight = {
  patient_name?: string | null;
  patient_id?: string | null;
  image_name?: string | null;
  probability: number;
  prediction: number;
  pathology?: string | null;
  notes?: string | null;
  created_at?: string | null;
};

const markdownComponents = {
  h1: (props: ComponentProps<"h4">) => (
    <h4 className="mt-2 text-xs font-semibold text-foreground" {...props} />
  ),
  h2: (props: ComponentProps<"h4">) => (
    <h4 className="mt-2 text-xs font-semibold text-foreground" {...props} />
  ),
  h3: (props: ComponentProps<"h4">) => (
    <h4 className="mt-2 text-xs font-semibold text-foreground" {...props} />
  ),
  p: (props: ComponentProps<"p">) => (
    <p className="text-xs leading-relaxed text-foreground/90" {...props} />
  ),
  ul: (props: ComponentProps<"ul">) => (
    <ul className="list-disc space-y-0.5 pl-4 text-xs" {...props} />
  ),
  ol: (props: ComponentProps<"ol">) => (
    <ol className="list-decimal space-y-0.5 pl-4 text-xs" {...props} />
  ),
  li: (props: ComponentProps<"li">) => <li className="text-foreground/90" {...props} />,
  strong: (props: ComponentProps<"strong">) => (
    <strong className="font-semibold text-foreground" {...props} />
  ),
  code: (props: ComponentProps<"code">) => (
    <code className="rounded bg-muted px-1 py-0.5 font-mono text-[10px]" {...props} />
  ),
};

type Props = {
  scan: ScanForInsight;
  disabled?: boolean;
};

export function ScanAiInsight({ scan, disabled }: Props) {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    setSummary(null);
    try {
      const { data, error: fnError } = await supabase.functions.invoke("summarize-scan", {
        body: { scan },
      });
      if (fnError) {
        let serverMsg: string | null = null;
        try {
          const ctx = (fnError as { context?: Response }).context;
          if (ctx && typeof ctx.json === "function") {
            const body = await ctx.json();
            serverMsg = body?.error ?? null;
          }
        } catch {
          // ignore
        }
        throw new Error(serverMsg ?? fnError.message);
      }
      const content = (data as { summary?: string } | null)?.summary;
      if (!content) throw new Error("Empty response from AI service.");
      setSummary(content);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to generate insight.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2 rounded-md border border-border bg-muted/30 p-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span
            className="inline-flex h-5 w-5 items-center justify-center rounded-full"
            style={{
              background: "var(--gradient-primary)",
              color: "var(--primary-foreground)",
            }}
            aria-hidden="true"
          >
            <Sparkles className="h-3 w-3" />
          </span>
          <span className="text-xs font-semibold">AI insight</span>
        </div>
        <div className="flex items-center gap-1">
          {(summary || error) && !loading && (
            <button
              type="button"
              onClick={() => {
                setSummary(null);
                setError(null);
              }}
              className="rounded-full p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Dismiss insight"
            >
              <X className="h-3 w-3" />
            </button>
          )}
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-7 gap-1 px-2 text-xs"
            onClick={run}
            disabled={loading || disabled}
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Sparkles className="h-3 w-3" />
            )}
            {loading ? "Thinking…" : summary ? "Regenerate" : "Summarize"}
          </Button>
        </div>
      </div>

      {loading && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Reviewing this scan…
        </p>
      )}
      {error && !loading && <p className="text-xs text-destructive">{error}</p>}
      {summary && !loading && (
        <div className="space-y-1.5">
          <ReactMarkdown components={markdownComponents}>{summary}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}
