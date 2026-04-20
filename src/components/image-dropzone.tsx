import { useCallback, useState } from "react";
import { UploadCloud, ImagePlus } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 15 * 1024 * 1024; // 15 MB

type Props = {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
};

export function ImageDropzone({ onFiles, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback((files: File[]) => {
    const valid: File[] = [];
    for (const f of files) {
      if (!ACCEPTED.includes(f.type)) {
        setError(`Unsupported file type: ${f.name}. Use JPG, PNG, or WebP.`);
        continue;
      }
      if (f.size > MAX_BYTES) {
        setError(`${f.name} is larger than 15 MB.`);
        continue;
      }
      valid.push(f);
    }
    return valid;
  }, []);

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    setError(null);
    const valid = validate(Array.from(fileList));
    if (valid.length > 0) onFiles(valid);
  };

  return (
    <div className="space-y-2">
      <label
        htmlFor="xray-input"
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (disabled) return;
          handleFiles(e.dataTransfer.files);
        }}
        className={cn(
          "group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed bg-card px-6 py-14 text-center transition-all",
          "border-border hover:border-primary/60 hover:bg-accent/40",
          dragging && "border-primary bg-accent/50 scale-[1.01]",
          disabled && "pointer-events-none opacity-60",
        )}
      >
        <span
          className="mb-4 flex h-14 w-14 items-center justify-center rounded-full text-primary-foreground shadow-[var(--shadow-elegant)] transition-transform group-hover:scale-110"
          style={{ background: "var(--gradient-primary)" }}
        >
          <UploadCloud className="h-7 w-7" />
        </span>
        <p className="text-base font-semibold text-foreground">
          Drop chest X-ray images here
        </p>
        <p className="mt-1 text-sm text-muted-foreground">
          or <span className="text-primary font-medium">click to browse</span> · JPG, PNG, WebP · up to 15 MB each
        </p>
        <p className="mt-3 inline-flex items-center gap-1.5 text-xs text-muted-foreground">
          <ImagePlus className="h-3.5 w-3.5" />
          Single or multiple images supported
        </p>

        <input
          id="xray-input"
          type="file"
          accept={ACCEPTED.join(",")}
          multiple
          className="sr-only"
          disabled={disabled}
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </label>

      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
