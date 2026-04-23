import { useEffect, useState } from "react";
import {
  Trash2,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  FileDown,
  Cloud,
  CloudOff,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";
import { PATHOLOGY_LABEL, type ClassificationResult } from "@/lib/classifier";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { ScanAiInsight } from "@/components/scan-ai-insight";
import { useAuth } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";
import { generateReport, safeFilenameFragment, type ReportPatient } from "@/lib/pdf-report";

export type AnalysisStatus = "pending" | "validating" | "analyzing" | "done" | "error";
export type SaveStatus = "idle" | "saving" | "saved" | "error";

export type AnalysisItem = {
  id: string;
  file: File;
  previewUrl: string;
  status: AnalysisStatus;
  result?: ClassificationResult;
  error?: string;
  notes: string;
  patientName: string;
  patientId: string;
  saveStatus: SaveStatus;
  saveError?: string;
  savedRecordId?: string;
};

type ResultsGalleryProps = {
  items: AnalysisItem[];
  onRemove: (id: string) => void;
  onClear: () => void;
  onNotesChange: (id: string, notes: string) => void;
  onPatientChange: (id: string, fields: { patientName?: string; patientId?: string }) => void;
};

export function ResultsGallery({
  items,
  onRemove,
  onClear,
  onNotesChange,
  onPatientChange,
}: ResultsGalleryProps) {
  if (items.length === 0) return null;

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Results</h2>
        <Button variant="outline" size="sm" onClick={onClear}>
          Clear all
        </Button>
      </div>
      <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
        <span className="font-medium text-foreground">Disclaimer.</span> This tool does not replace
        professional medical diagnosis. Model limitations: placeholder for population coverage, edge
        cases, and known failure modes.
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((item) => (
          <article
            key={item.id}
            className="overflow-hidden rounded-xl border border-border bg-card"
            style={{ boxShadow: "var(--shadow-card)" }}
          >
            <div className="space-y-3 p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 space-y-0.5">
                  <p className="line-clamp-2 text-sm font-medium">{item.file.name}</p>
                  <SaveIndicator item={item} />
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => onRemove(item.id)}
                  aria-label={`Remove ${item.file.name}`}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
              <PatientFields item={item} onPatientChange={onPatientChange} />
            </div>
            <div className="aspect-[4/3] w-full bg-muted">
              <img
                src={item.previewUrl}
                alt={item.file.name}
                className="h-full w-full object-cover"
              />
            </div>
            <div className="space-y-3 p-4 pt-5">
              <ResultState item={item} onNotesChange={onNotesChange} />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
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

async function fetchAiSummary(item: AnalysisItem): Promise<string | null> {
  if (!item.result) return null;
  try {
    const { data, error } = await supabase.functions.invoke("summarize-scan", {
      body: {
        scan: {
          patient_name: item.patientName || null,
          patient_id: item.patientId || null,
          image_name: item.file.name,
          probability: item.result.probability,
          prediction: item.result.prediction,
          pathology: item.result.label || PATHOLOGY_LABEL,
          notes: item.notes || null,
        },
      },
    });
    if (error) throw error;
    const content = (data as { summary?: string } | null)?.summary;
    return content ?? null;
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Failed to generate AI summary.";
    toast.error(msg);
    return null;
  }
}

async function downloadPdf(
  item: AnalysisItem,
  info: ReportPatient,
  includeAi: boolean,
): Promise<void> {
  if (!item.result) return;

  const aiSummary = includeAi ? await fetchAiSummary(item) : null;

  const doc = await generateReport({
    patient: info,
    scan: {
      imageName: item.file.name,
      imageUrl: item.previewUrl,
      probability: item.result.probability,
      prediction: item.result.prediction,
      pathology: item.result.label || PATHOLOGY_LABEL,
      notes: item.notes,
    },
    aiSummary,
  });

  const patientFragment = safeFilenameFragment(info.patientName, "patient");
  const idFragment = safeFilenameFragment(info.patientId, "id");
  const baseName = item.file.name.replace(/\.[^.]+$/, "");
  doc.save(`${patientFragment}-${idFragment}-${baseName}-report.pdf`);
}

function ResultState({
  item,
  onNotesChange,
}: {
  item: AnalysisItem;
  onNotesChange: (id: string, notes: string) => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);

  const onGenerate = async (info: ReportPatient) => {
    setDownloading(true);
    try {
      await downloadPdf(item, info);
      setReportOpen(false);
    } finally {
      setDownloading(false);
    }
  };

  const sharedNotesProps = {
    value: item.notes,
    onChange: (notes: string) => onNotesChange(item.id, notes),
    onDownload: () => setReportOpen(true),
    downloading,
  };

  const dialog = (
    <ReportDialog
      open={reportOpen}
      onOpenChange={setReportOpen}
      onGenerate={onGenerate}
      submitting={downloading}
      canGenerate={Boolean(item.result)}
      defaultPatientName={item.patientName}
      defaultPatientId={item.patientId}
    />
  );

  if (item.status === "pending" || item.status === "validating" || item.status === "analyzing") {
    const statusLabel =
      item.status === "pending"
        ? "Queued..."
        : item.status === "validating"
          ? "Checking if chest X-ray..."
          : "Analyzing...";
    return (
      <div className="space-y-3">
        <p className="inline-flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          {statusLabel}
        </p>
        <NotesBlock {...sharedNotesProps} />
        {dialog}
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
        <NotesBlock {...sharedNotesProps} />
        {dialog}
      </div>
    );
  }

  if (!item.result) {
    return (
      <div className="space-y-3">
        <p className="text-sm text-muted-foreground">No result available.</p>
        <NotesBlock {...sharedNotesProps} />
        {dialog}
      </div>
    );
  }

  const probabilityPct = `${(item.result.probability * 100).toFixed(1)}%`;
  const verdict = item.result.prediction === 1 ? "Potential indication" : "No clear indication";
  const verdictClass = item.result.prediction === 1 ? "text-amber-600" : "text-emerald-600";
  const lowConfidence = item.result.probability < 0.9;

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
          Confidence (~90%). Interpret with caution and specialist review.
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
      <ScanAiInsight
        scan={{
          patient_name: item.patientName || null,
          patient_id: item.patientId || null,
          image_name: item.file.name,
          probability: item.result.probability,
          prediction: item.result.prediction,
          pathology: "cardiomegaly",
          notes: item.notes || null,
        }}
      />
      <NotesBlock {...sharedNotesProps} />
      <ScanAiInsight
        scan={{
          patient_name: item.patientName,
          patient_id: item.patientId,
          image_name: item.file.name,
          probability: item.result.probability,
          prediction: item.result.prediction,
          pathology: item.result.label || PATHOLOGY_LABEL,
          notes: item.notes,
        }}
      />
      {dialog}
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
  onDownload: () => void;
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
      <Button size="sm" variant="outline" onClick={onDownload} disabled={downloading}>
        <FileDown className="h-4 w-4" />
        {downloading ? "Generating PDF..." : "Download PDF"}
      </Button>
    </div>
  );
}

function ReportDialog({
  open,
  onOpenChange,
  onGenerate,
  submitting,
  canGenerate,
  defaultPatientName,
  defaultPatientId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onGenerate: (info: ReportPatient) => Promise<void> | void;
  submitting: boolean;
  canGenerate: boolean;
  defaultPatientName?: string;
  defaultPatientId?: string;
}) {
  const { user } = useAuth();
  const profileName = (user?.user_metadata?.full_name as string | undefined) ?? "";
  const profileEmail = user?.email ?? "";
  const fallbackDoctor = profileName || (profileEmail ? profileEmail.split("@")[0] : "");

  const [doctorName, setDoctorName] = useState(fallbackDoctor);
  const [patientName, setPatientName] = useState(defaultPatientName ?? "");
  const [patientId, setPatientId] = useState(defaultPatientId ?? "");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setDoctorName(fallbackDoctor);
      setPatientName(defaultPatientName ?? "");
      setPatientId(defaultPatientId ?? "");
      setError(null);
    }
  }, [open, fallbackDoctor, defaultPatientName, defaultPatientId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canGenerate) {
      setError("No analysis result yet. Wait for the prediction to finish.");
      return;
    }
    if (patientName.trim().length === 0) {
      setError("Patient name is required.");
      return;
    }
    if (patientId.trim().length === 0) {
      setError("Patient ID is required.");
      return;
    }
    if (doctorName.trim().length === 0) {
      setError("Doctor name is required.");
      return;
    }
    setError(null);
    await onGenerate({
      doctorName: doctorName.trim(),
      patientName: patientName.trim(),
      patientId: patientId.trim(),
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Generate patient report</DialogTitle>
          <DialogDescription>
            Confirm the doctor and patient names that will appear on the PDF.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="report-patient">Patient name</Label>
              <Input
                id="report-patient"
                autoFocus
                required
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="e.g. John Doe"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="report-patient-id">Patient ID</Label>
              <Input
                id="report-patient-id"
                required
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g. MRN-00482"
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="report-doctor">Doctor</Label>
            <Input
              id="report-doctor"
              required
              value={doctorName}
              onChange={(e) => setDoctorName(e.target.value)}
              placeholder="Dr. Jane Smith"
            />
            {!profileName && (
              <p className="text-xs text-muted-foreground">
                Tip: set this once on your profile to skip the field next time.
              </p>
            )}
          </div>
          {error && (
            <p className="rounded-md border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={submitting || !canGenerate}>
              {submitting ? "Generating..." : "Generate PDF"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function PatientFields({
  item,
  onPatientChange,
}: {
  item: AnalysisItem;
  onPatientChange: (id: string, fields: { patientName?: string; patientId?: string }) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="space-y-1">
        <Label htmlFor={`patient-name-${item.id}`} className="text-xs text-muted-foreground">
          Patient name
        </Label>
        <Input
          id={`patient-name-${item.id}`}
          value={item.patientName}
          onChange={(e) => onPatientChange(item.id, { patientName: e.target.value })}
          placeholder="e.g. John Doe"
          className="h-8 text-xs"
        />
      </div>
      <div className="space-y-1">
        <Label htmlFor={`patient-id-${item.id}`} className="text-xs text-muted-foreground">
          Patient ID
        </Label>
        <Input
          id={`patient-id-${item.id}`}
          value={item.patientId}
          onChange={(e) => onPatientChange(item.id, { patientId: e.target.value })}
          placeholder="MRN-00482"
          className="h-8 text-xs"
        />
      </div>
    </div>
  );
}

function SaveIndicator({ item }: { item: AnalysisItem }) {
  if (item.saveStatus === "saving") {
    return (
      <p className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        Saving to history...
      </p>
    );
  }
  if (item.saveStatus === "saved") {
    return (
      <p className="inline-flex items-center gap-1 text-xs text-emerald-600">
        <Cloud className="h-3 w-3" />
        Saved to history
      </p>
    );
  }
  if (item.saveStatus === "error") {
    return (
      <p
        className="inline-flex items-center gap-1 text-xs text-amber-600"
        title={item.saveError ?? "Could not save"}
      >
        <CloudOff className="h-3 w-3" />
        Not saved
      </p>
    );
  }
  return null;
}
