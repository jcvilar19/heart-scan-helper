import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Loader2, Save, UserRound } from "lucide-react";
import { AppHeader } from "@/components/app-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/profile")({
  component: ProfilePage,
  head: () => ({
    meta: [
      { title: "Profile — CardioScan" },
      {
        name: "description",
        content: "Manage the doctor name shown on patient reports.",
      },
    ],
  }),
});

function ProfilePage() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  const [fullName, setFullName] = useState("");
  const [saving, setSaving] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      navigate({ to: "/auth" });
    }
  }, [loading, user, navigate]);

  useEffect(() => {
    if (user) {
      const stored = (user.user_metadata?.full_name as string | undefined) ?? "";
      setFullName(stored);
      setHydrated(true);
    }
  }, [user]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = fullName.trim();
    if (trimmed.length < 2) {
      toast.error("Enter your full name (at least 2 characters).");
      return;
    }
    setSaving(true);
    try {
      const { error } = await supabase.auth.updateUser({
        data: { full_name: trimmed },
      });
      if (error) throw error;
      toast.success("Profile updated.");
    } catch (err) {
      toast.error((err as Error).message ?? "Could not update profile.");
    } finally {
      setSaving(false);
    }
  };

  if (loading || !user) {
    return (
      <div className="min-h-screen bg-background">
        <AppHeader />
        <main className="mx-auto flex min-h-[60vh] max-w-md items-center justify-center px-4">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <AppHeader />
      <main className="mx-auto max-w-xl px-4 py-10">
        <div className="mb-6 flex items-center gap-3">
          <span
            className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted text-foreground"
          >
            <UserRound className="h-5 w-5" />
          </span>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Profile</h1>
            <p className="text-sm text-muted-foreground">
              This name appears on every patient report you generate.
            </p>
          </div>
        </div>

        <div
          className="rounded-2xl border border-border bg-card p-6"
          style={{ boxShadow: "var(--shadow-card)" }}
        >
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="profile-email">Email</Label>
              <Input id="profile-email" value={user.email ?? ""} disabled readOnly />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="profile-name">Doctor’s full name</Label>
              <Input
                id="profile-name"
                autoComplete="name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="e.g. Dr. Jane Smith"
                disabled={!hydrated}
              />
              <p className="text-xs text-muted-foreground">
                Shown on patient reports as the signing doctor.
              </p>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <Button asChild variant="ghost" size="sm">
                <Link to="/app">Back to scanner</Link>
              </Button>
              <Button type="submit" size="sm" disabled={saving || !hydrated}>
                <Save className="h-4 w-4" />
                {saving ? "Saving…" : "Save changes"}
              </Button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
