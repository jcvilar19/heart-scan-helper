import { useState, type ComponentProps } from "react";
import { Loader2, Sparkles, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";

export type ScanInsightInput = {
  patient_name?: string | null;
  patient_id?: string | null;
  image_name?: string | null;
  probability?: number | null;
  prediction?: number | null;
  pathology?: string | null;
  notes?: string | null;
  created_at?: string | null;
};

const markdownComponents = {
  h1: (props: ComponentProps<"h4">) => <h4 className="mt-3 text-sm font-semibold" {...props} />,
  h2: (props: ComponentProps<"h4">) => <h4 className="mt-3 text-sm font-semibold" {...props} />,
  h3: (props: ComponentProps<"h4">) => <h4 className="mt-3 text-sm font-semibold" {...props} />,
  p: (props: ComponentProps<"p">) => <p className="text-xs text-foreground/90" {...props} />,
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

type ScanAiInsightProps = {
  scan: ScanInsightInput;
  disabled?: boolean;
};

export function ScanAiInsight({ scan, disabled }: ScanAiInsightProps) {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    setSummary(null);
    try {
      const { data, error: invokeError } = await supabase.functions.invoke("summarize-scan", {
        body: { scan },
      });
      if (invokeError) {
        let serverMsg: string | null = null;
        try {
          const ctx = (invokeError as { context?: Response }).context;
          if (ctx && typeof ctx.json === "function") {
            const body = await ctx.json();
            serverMsg = body?.error ?? null;
          }
        } catch {
          // ignore
        }
        throw new Error(serverMsg ?? invokeError.message);
      }
      const text = (data as { summary?: string } | null)?.summary;
      if (!text) throw new Error("Empty response from AI service.");
      setSummary(text);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to generate insight.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2 rounded-md border border-border p-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span
            className="inline-flex h-5 w-5 items-center justify-center rounded-full"
            style={{
              background: "var(--gradient-primary)",
              color: "var(--primary-foreground)",
            }}
          >
            <Sparkles className="h-3 w-3" />
          </span>
          <span className="text-xs font-medium text-foreground">AI insight</span>
        </div>
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
      </div>

      {!summary && !loading && !error && (
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={run}
          disabled={disabled || loading}
          className="w-full gap-1.5"
        >
          <Sparkles className="h-3.5 w-3.5" />
          AI summarize
        </Button>
      )}

      {loading && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Generating insight…
        </div>
      )}

      {error && !loading && (
        <div className="space-y-2">
          <p className="text-xs text-destructive">{error}</p>
          <Button type="button" size="sm" variant="outline" onClick={run} className="w-full">
            Retry
          </Button>
        </div>
      )}

      {summary && !loading && (
        <div className="space-y-1 leading-relaxed">
          <ReactMarkdown components={markdownComponents}>{summary}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}
