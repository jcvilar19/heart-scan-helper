import { useEffect, useState } from "react";
import { FileDown, Pencil, Save, Sparkles, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useAuth } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";
import { generateReport, safeFilenameFragment, type ReportPatient } from "@/lib/pdf-report";
import { PATHOLOGY_LABEL } from "@/lib/classifier";

export type HistoryRow = {
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

type Props = {
  row: HistoryRow;
  onUpdate: (id: string, patch: Partial<HistoryRow>) => void;
};

export function HistoryCardTools({ row, onUpdate }: Props) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(row.patient_name ?? "");
  const [pid, setPid] = useState(row.patient_id ?? "");
  const [saving, setSaving] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (!editing) {
      setName(row.patient_name ?? "");
      setPid(row.patient_id ?? "");
    }
  }, [row.patient_name, row.patient_id, editing]);

  const handleSave = async () => {
    setSaving(true);
    const patch = {
      patient_name: name.trim() || null,
      patient_id: pid.trim() || null,
    };
    const { error } = await supabase.from("classifications").update(patch).eq("id", row.id);
    setSaving(false);
    if (error) {
      toast.error(error.message);
      return;
    }
    onUpdate(row.id, patch);
    setEditing(false);
    toast.success("Patient updated.");
  };

  const handleGenerate = async (info: ReportPatient, includeAi: boolean) => {
    setDownloading(true);
    try {
      let aiSummary: string | null = null;
      if (includeAi) {
        try {
          const { data, error } = await supabase.functions.invoke("summarize-scan", {
            body: {
              scan: {
                patient_name: info.patientName,
                patient_id: info.patientId,
                image_name: row.image_name,
                probability: row.probability,
                prediction: row.prediction,
                pathology: row.pathology || PATHOLOGY_LABEL,
                notes: row.notes,
              },
            },
          });
          if (error) throw error;
          aiSummary = (data as { summary?: string } | null)?.summary ?? null;
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Failed to generate AI summary.";
          toast.error(msg);
        }
      }

      const doc = await generateReport({
        patient: info,
        scan: {
          imageName: row.image_name,
          imageUrl: row.signedUrl ?? "",
          probability: row.probability,
          prediction: row.prediction,
          pathology: row.pathology || PATHOLOGY_LABEL,
          notes: row.notes,
          createdAt: row.created_at,
        },
        aiSummary,
      });

      const patientFragment = safeFilenameFragment(info.patientName, "patient");
      const idFragment = safeFilenameFragment(info.patientId, "id");
      const baseName = row.image_name.replace(/\.[^.]+$/, "");
      doc.save(`${patientFragment}-${idFragment}-${baseName}-report.pdf`);
      setReportOpen(false);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-2 border-t border-border pt-3">
      {editing ? (
        <div className="space-y-2">
          <div className="space-y-1">
            <Label htmlFor={`edit-name-${row.id}`} className="text-xs text-muted-foreground">
              Patient name
            </Label>
            <Input
              id={`edit-name-${row.id}`}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. John Doe"
              className="h-8 text-xs"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor={`edit-id-${row.id}`} className="text-xs text-muted-foreground">
              Patient ID
            </Label>
            <Input
              id={`edit-id-${row.id}`}
              value={pid}
              onChange={(e) => setPid(e.target.value)}
              placeholder="MRN-00482"
              className="h-8 text-xs"
            />
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={handleSave} disabled={saving} className="flex-1">
              <Save className="h-3.5 w-3.5" />
              {saving ? "Saving..." : "Save"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setEditing(false)}
              disabled={saving}
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setEditing(true)}
            className="flex-1"
          >
            <Pencil className="h-3.5 w-3.5" />
            Edit
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setReportOpen(true)}
            disabled={downloading}
            className="flex-1"
          >
            <FileDown className="h-3.5 w-3.5" />
            {downloading ? "Generating..." : "PDF"}
          </Button>
        </div>
      )}

      <ReportDialog
        open={reportOpen}
        onOpenChange={setReportOpen}
        onGenerate={handleGenerate}
        submitting={downloading}
        defaultPatientName={row.patient_name ?? ""}
        defaultPatientId={row.patient_id ?? ""}
      />
    </div>
  );
}

function ReportDialog({
  open,
  onOpenChange,
  onGenerate,
  submitting,
  defaultPatientName,
  defaultPatientId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onGenerate: (info: ReportPatient, includeAi: boolean) => Promise<void> | void;
  submitting: boolean;
  defaultPatientName: string;
  defaultPatientId: string;
}) {
  const { user } = useAuth();
  const profileName = (user?.user_metadata?.full_name as string | undefined) ?? "";
  const profileEmail = user?.email ?? "";
  const fallbackDoctor = profileName || (profileEmail ? profileEmail.split("@")[0] : "");

  const [doctorName, setDoctorName] = useState(fallbackDoctor);
  const [patientName, setPatientName] = useState(defaultPatientName);
  const [patientId, setPatientId] = useState(defaultPatientId);
  const [includeAi, setIncludeAi] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setDoctorName(fallbackDoctor);
      setPatientName(defaultPatientName);
      setPatientId(defaultPatientId);
      setIncludeAi(true);
      setError(null);
    }
  }, [open, fallbackDoctor, defaultPatientName, defaultPatientId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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
    await onGenerate(
      {
        doctorName: doctorName.trim(),
        patientName: patientName.trim(),
        patientId: patientId.trim(),
      },
      includeAi,
    );
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
              <Label htmlFor="hist-report-patient">Patient name</Label>
              <Input
                id="hist-report-patient"
                autoFocus
                required
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="e.g. John Doe"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="hist-report-patient-id">Patient ID</Label>
              <Input
                id="hist-report-patient-id"
                required
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g. MRN-00482"
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hist-report-doctor">Doctor</Label>
            <Input
              id="hist-report-doctor"
              required
              value={doctorName}
              onChange={(e) => setDoctorName(e.target.value)}
              placeholder="Dr. Jane Smith"
            />
          </div>
          <div
            className="flex items-start gap-3 rounded-md border border-border p-3"
            style={{ background: "color-mix(in oklab, var(--primary) 4%, transparent)" }}
          >
            <Checkbox
              id="hist-report-include-ai"
              checked={includeAi}
              onCheckedChange={(v) => setIncludeAi(v === true)}
              className="mt-0.5"
            />
            <div className="flex-1 space-y-0.5">
              <Label
                htmlFor="hist-report-include-ai"
                className="flex cursor-pointer items-center gap-1.5 text-sm font-medium"
              >
                <Sparkles className="h-3.5 w-3.5 text-primary" />
                Include AI clinical summary
              </Label>
              <p className="text-xs text-muted-foreground">
                Adds a dedicated page with an AI-generated case summary.
              </p>
            </div>
          </div>
          {error && (
            <p className="rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1 text-xs text-destructive">
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
            <Button type="submit" disabled={submitting}>
              {submitting ? "Generating..." : "Generate PDF"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
