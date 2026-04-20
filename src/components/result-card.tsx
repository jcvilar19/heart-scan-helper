import { useState } from "react";
import { jsPDF } from "jspdf";
import { Trash2, Loader2, AlertTriangle, CheckCircle2, FileDown } from "lucide-react";
import { PATHOLOGY_LABEL, type ClassificationResult } from "@/lib/classifier";
import { Button } from "@/components/ui/button";

export type AnalysisStatus = "pending" | "analyzing" | "done" | "error";

export type AnalysisItem = {
  id: string;
  file: File;
  previewUrl: string;
  status: AnalysisStatus;
  result?: ClassificationResult;
  error?: string;
  notes: string;
};

type ResultsGalleryProps = {
  items: AnalysisItem[];
  onRemove: (id: string) => void;
  onClear: () => void;
  onNotesChange: (id: string, notes: string) => void;
};

export function ResultsGallery({ items, onRemove, onClear, onNotesChange }: ResultsGalleryProps) {
  if (items.length === 0) return null;

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Results</h2>
        <Button variant="outline" size="sm" onClick={onClear}>
          Clear all
        </Button>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((item) => (
          <article
            key={item.id}
            className="overflow-hidden rounded-xl border border-border bg-card"
            style={{ boxShadow: "var(--shadow-card)" }}
          >
            <div className="aspect-[4/3] w-full bg-muted">
              <img
                src={item.previewUrl}
                alt={item.file.name}
                className="h-full w-full object-cover"
              />
            </div>
            <div className="space-y-3 p-4">
              <div className="flex items-start justify-between gap-3">
                <p className="line-clamp-2 text-sm font-medium">{item.file.name}</p>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => onRemove(item.id)}
                  aria-label={`Remove ${item.file.name}`}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
              <ResultState item={item} onNotesChange={onNotesChange} />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

async function imageToDataUrl(src: string): Promise<string> {
  return await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Could not create canvas context."));
        return;
      }
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL("image/jpeg", 0.95));
    };
    img.onerror = () => reject(new Error("Could not load image for PDF."));
    img.src = src;
  });
}

function predictionSummary(item: AnalysisItem): string {
  if (item.result) {
    return item.result.label || PATHOLOGY_LABEL;
  }
  if (item.status === "error") {
    return "Prediction failed";
  }
  if (item.status === "analyzing") {
    return "Analyzing";
  }
  return "Pending";
}

async function downloadPdf(item: AnalysisItem): Promise<void> {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 14;
  let y = 16;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.text("Medical Image Report", margin, y);
  y += 10;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.text(`Image name: ${item.file.name}`, margin, y);
  y += 7;
  doc.text(`Prediction: ${predictionSummary(item)}`, margin, y);
  y += 7;

  const confidence = item.result ? `${(item.result.probability * 100).toFixed(1)}%` : "N/A";
  doc.text(`Confidence: ${confidence}`, margin, y);
  y += 8;

  if (item.notes.trim().length > 0) {
    doc.setFont("helvetica", "bold");
    doc.text("Clinical notes:", margin, y);
    y += 6;
    doc.setFont("helvetica", "normal");
    const wrapped = doc.splitTextToSize(item.notes.trim(), pageWidth - margin * 2);
    doc.text(wrapped, margin, y);
    y += wrapped.length * 5 + 4;
  }

  const dataUrl = await imageToDataUrl(item.previewUrl);
  const maxWidth = pageWidth - margin * 2;
  const maxHeight = 140;
  doc.addImage(dataUrl, "JPEG", margin, y, maxWidth, maxHeight, undefined, "FAST");

  const filename = `${item.file.name.replace(/\.[^.]+$/, "")}-report.pdf`;
  doc.save(filename);
}

function ResultState({
  item,
  onNotesChange,
}: {
  item: AnalysisItem;
  onNotesChange: (id: string, notes: string) => void;
}) {
  const [downloading, setDownloading] = useState(false);

  if (item.status === "pending" || item.status === "analyzing") {
    return (
      <div className="space-y-3">
        <p className="inline-flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          {item.status === "pending" ? "Queued..." : "Analyzing..."}
        </p>
        <NotesBlock
          value={item.notes}
          onChange={(notes) => onNotesChange(item.id, notes)}
          onDownload={async () => {
            setDownloading(true);
            try {
              await downloadPdf(item);
            } finally {
              setDownloading(false);
            }
          }}
          downloading={downloading}
        />
      </div>
    );
  }

  if (item.status === "error") {
    return (
      <div className="space-y-3">
        <p className="inline-flex items-center gap-2 text-sm text-destructive">
          <AlertTriangle className="h-4 w-4" />
          {item.error ?? "Could not process this image."}
        </p>
        <NotesBlock
          value={item.notes}
          onChange={(notes) => onNotesChange(item.id, notes)}
          onDownload={async () => {
            setDownloading(true);
            try {
              await downloadPdf(item);
            } finally {
              setDownloading(false);
            }
          }}
          downloading={downloading}
        />
      </div>
    );
  }

  if (!item.result) {
    return (
      <div className="space-y-3">
        <p className="text-sm text-muted-foreground">No result available.</p>
        <NotesBlock
          value={item.notes}
          onChange={(notes) => onNotesChange(item.id, notes)}
          onDownload={async () => {
            setDownloading(true);
            try {
              await downloadPdf(item);
            } finally {
              setDownloading(false);
            }
          }}
          downloading={downloading}
        />
      </div>
    );
  }

  const probabilityPct = `${(item.result.probability * 100).toFixed(1)}%`;
  const verdict = item.result.prediction === 1 ? "Potential indication" : "No clear indication";
  const verdictClass = item.result.prediction === 1 ? "text-amber-600" : "text-emerald-600";
  const lowConfidence = item.result.probability < 0.7;

  return (
    <div className="space-y-2 text-sm">
      <p className="inline-flex items-center gap-2 text-foreground">
        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
        Analysis complete
      </p>
      <p className="text-muted-foreground">
        Model suggests:{" "}
        <span className="font-medium text-foreground">{item.result.label || PATHOLOGY_LABEL}</span>
      </p>
      <p className="text-muted-foreground">
        Confidence: <span className="font-medium text-foreground">{probabilityPct}</span>
      </p>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={item.result.prediction === 1 ? "h-full bg-amber-500" : "h-full bg-emerald-500"}
          style={{ width: probabilityPct }}
        />
      </div>
      <p className="text-muted-foreground text-xs">
        Verdict: <span className={`font-medium ${verdictClass}`}>{verdict}</span>
      </p>
      {lowConfidence && (
        <p className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-xs text-amber-700">
          Low confidence (&lt;70%). Interpret with caution and specialist review.
        </p>
      )}
      {item.result.heatmapUrl && (
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">Grad-CAM heatmap</p>
          <img
            src={item.result.heatmapUrl}
            alt={`Heatmap for ${item.file.name}`}
            className="max-h-52 w-full rounded-md border border-border object-cover"
          />
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        This tool does not replace professional medical diagnosis.
      </p>
      <p className="text-xs text-muted-foreground">
        Model limitations: placeholder for population coverage, edge cases, and known failure modes.
      </p>
      <NotesBlock
        value={item.notes}
        onChange={(notes) => onNotesChange(item.id, notes)}
        onDownload={async () => {
          setDownloading(true);
          try {
            await downloadPdf(item);
          } finally {
            setDownloading(false);
          }
        }}
        downloading={downloading}
      />
    </div>
  );
}

function NotesBlock({
  value,
  onChange,
  onDownload,
  downloading,
}: {
  value: string;
  onChange: (notes: string) => void;
  onDownload: () => Promise<void>;
  downloading: boolean;
}) {
  return (
    <div className="space-y-2 rounded-md border border-border p-2">
      <label className="text-xs font-medium text-foreground">Clinical notes</label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Write medical notes for this image..."
        className="min-h-20 w-full resize-y rounded-md border border-border bg-background px-2 py-1 text-xs outline-none focus:border-primary"
      />
      <Button size="sm" variant="outline" onClick={() => void onDownload()} disabled={downloading}>
        <FileDown className="h-4 w-4" />
        {downloading ? "Generating PDF..." : "Download PDF"}
      </Button>
    </div>
  );
}
