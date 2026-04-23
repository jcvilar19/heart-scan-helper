import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import {
  ArrowRight,
  Brain,
  ShieldCheck,
  Zap,
  
  Upload,
  LineChart,
  Stethoscope,
} from "lucide-react";
import { AppHeader } from "@/components/app-header";
import { Button } from "@/components/ui/button";
import { NeuralNetworkAnim } from "@/components/neural-network-anim";
import { useAuth } from "@/hooks/use-auth";

export const Route = createFileRoute("/")({
  component: LandingPage,
  head: () => ({
    meta: [
      { title: "Coraçai — AI Cardiomegaly Detection from Chest X-rays" },
      {
        name: "description",
        content:
          "AI-assisted cardiomegaly screening from chest X-rays. Upload images and get a probability score in seconds.",
      },
      { property: "og:title", content: "Coraçai — AI Cardiomegaly Detection" },
      {
        property: "og:description",
        content: "AI-assisted cardiomegaly screening from chest X-rays.",
      },
    ],
  }),
});

function LandingPage() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && user) {
      navigate({ to: "/app", replace: true });
    }
  }, [user, loading, navigate]);

  const ctaTo = user ? "/app" : "/auth";
  const ctaLabel = user ? "Open scanner" : "Sign in to start";

  if (!loading && user) return null;

  return (
    <div className="min-h-screen" style={{ background: "var(--gradient-surface)" }}>
      <AppHeader />

      {/* HERO */}
      <section className="relative overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute -top-32 right-[-10%] h-[480px] w-[480px] rounded-full opacity-30 blur-3xl"
          style={{ background: "var(--gradient-primary)" }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute -bottom-40 left-[-10%] h-[420px] w-[420px] rounded-full opacity-20 blur-3xl"
          style={{ background: "var(--gradient-primary)" }}
        />

        <div className="mx-auto grid max-w-6xl gap-12 px-4 pb-20 pt-16 sm:px-6 lg:grid-cols-2 lg:items-center lg:pt-24">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
              ML-assisted screening · Research preview
            </div>
            <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              Detect{" "}
              <span
                className="bg-clip-text text-transparent"
                style={{ backgroundImage: "var(--gradient-primary)" }}
              >
                Cardiomegaly
              </span>{" "}
              <br className="hidden sm:inline" />
              from X-Rays <br className="hidden sm:inline" />
              in seconds.
            </h1>
            <p className="mt-5 max-w-xl text-pretty text-base text-muted-foreground sm:text-lg">
              Coraçai analyzes chest radiography with a ML model trained on thousands of
              labelled images. Get a probability score and binary verdict the moment you upload.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Button asChild size="lg">
                <Link to={ctaTo}>
                  {ctaLabel}
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <a href="#how-it-works">How it works</a>
              </Button>
            </div>

            <dl className="mt-10 grid max-w-md grid-cols-3 gap-6">
              <Stat label="Inference" value="<2s" />
              <Stat label="Threshold" value="0.4" />
              <Stat label="Per-user data" value="Private" />
            </dl>
          </div>

          <div className="relative">
            <div
              className="rounded-3xl border border-border bg-card p-6"
              style={{ boxShadow: "var(--shadow-elegant)" }}
            >
              <NeuralNetworkAnim />
            </div>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="mx-auto max-w-6xl px-4 pb-20 sm:px-6">
        <div className="mx-auto mb-10 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Built for clinicians and researchers
          </h2>
          <p className="mt-3 text-muted-foreground">
            A focused workflow: upload, classify, review, and save — nothing in the way.
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          <FeatureTile
            icon={<Zap className="h-5 w-5" />}
            title="Instant inference"
            body="Probability score per image, returned in under two seconds."
          />
          <FeatureTile
            icon={<Brain className="h-5 w-5" />}
            title="Binary verdict"
            body="Threshold of 0.4 maps probability to a clear 0/1 prediction."
          />
          <FeatureTile
            icon={<ShieldCheck className="h-5 w-5" />}
            title="Private by default"
            body="Saved scans are protected by per-user access policies."
          />
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section
        id="how-it-works"
        className="border-y border-border/60"
        style={{ background: "color-mix(in oklab, var(--primary) 4%, var(--background))" }}
      >
        <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
          <div className="mx-auto mb-12 max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">How it works</h2>
            <p className="mt-3 text-muted-foreground">
              Three steps from upload to verdict.
            </p>
          </div>
          <ol className="grid gap-6 sm:grid-cols-3">
            <Step
              n={1}
              icon={<Upload className="h-5 w-5" />}
              title="Upload X-rays"
              body="Drag & drop one or many chest X-rays. JPG or PNG, any size."
            />
            <Step
              n={2}
              icon={<LineChart className="h-5 w-5" />}
              title="AI inference"
              body="Our model returns a calibrated probability score for each image."
            />
            <Step
              n={3}
              icon={<Stethoscope className="h-5 w-5" />}
              title="Review & save"
              body="Add notes, get a summary, save to your private history, and revisit any time."
            />
          </ol>
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <div
          className="overflow-hidden rounded-3xl border border-border p-10 text-center sm:p-14"
          style={{
            background: "var(--gradient-primary)",
            boxShadow: "var(--shadow-elegant)",
          }}
        >
          <h2 className="text-balance text-3xl font-bold tracking-tight text-primary-foreground sm:text-4xl">
            Ready to analyze your first scan?
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-pretty text-primary-foreground/85">
            {user
              ? "Jump into the scanner and upload an X-ray."
              : "Create a free account to start uploading X-rays and saving results."}
          </p>
          <div className="mt-8 flex justify-center">
            <Button
              asChild
              size="lg"
              variant="secondary"
              className="bg-card text-foreground hover:bg-card/90"
            >
              <Link to={ctaTo}>
                {ctaLabel}
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      <footer className="mx-auto max-w-6xl px-4 pb-10 text-center text-xs text-muted-foreground sm:px-6">
        For research and educational use only. Not a substitute for professional medical
        diagnosis.
      </footer>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-widest text-muted-foreground">{label}</dt>
      <dd className="mt-1 text-xl font-semibold tracking-tight">{value}</dd>
    </div>
  );
}

function FeatureTile({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div
      className="rounded-2xl border border-border bg-card p-6 transition-transform hover:-translate-y-0.5"
      style={{ boxShadow: "var(--shadow-card)" }}
    >
      <div className="mb-3 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-accent text-primary">
        {icon}
      </div>
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="mt-1.5 text-sm text-muted-foreground">{body}</p>
    </div>
  );
}

function Step({
  n,
  icon,
  title,
  body,
}: {
  n: number;
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <li
      className="relative rounded-2xl border border-border bg-card p-6 text-center"
      style={{ boxShadow: "var(--shadow-card)" }}
    >
      <span className="absolute left-3 top-3 inline-flex h-7 w-7 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
        {n}
      </span>
      <div className="mx-auto mb-3 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-accent text-primary">
        {icon}
      </div>
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="mt-1.5 text-sm text-muted-foreground">{body}</p>
    </li>
  );
}
