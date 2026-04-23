// Lovable AI–powered summary of a user's classification history.
// Receives a compact summary of recent scans and asks the model to
// produce a concise clinical-style summary + suggestions for improvement.

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

type ScanInput = {
  patient_name?: string | null;
  patient_id?: string | null;
  image_name?: string | null;
  probability: number;
  prediction: number;
  pathology?: string | null;
  notes?: string | null;
  created_at?: string | null;
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { scans } = (await req.json()) as { scans: ScanInput[] };
    if (!Array.isArray(scans) || scans.length === 0) {
      return new Response(JSON.stringify({ error: "No scans provided." }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      return new Response(JSON.stringify({ error: "LOVABLE_API_KEY not configured" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Compact, model-friendly representation of each scan.
    const lines = scans.slice(0, 100).map((s, i) => {
      const verdict = s.prediction === 1 ? "POSITIVE" : "negative";
      const prob = (s.probability * 100).toFixed(1);
      const date = s.created_at ? new Date(s.created_at).toISOString().slice(0, 10) : "—";
      const who =
        [s.patient_name, s.patient_id ? `#${s.patient_id}` : null].filter(Boolean).join(" ") ||
        s.image_name ||
        "unknown";
      const notes = s.notes ? ` notes: ${s.notes.slice(0, 200)}` : "";
      return `${i + 1}. ${date} | ${who} | ${verdict} (${prob}%)${notes}`;
    });

    const total = scans.length;
    const positive = scans.filter((s) => s.prediction === 1).length;
    const avgProb =
      scans.reduce((acc, s) => acc + (s.probability ?? 0), 0) / Math.max(1, total);

    const systemPrompt =
      "You are a clinical AI assistant helping a clinician review a batch of cardiomegaly screening results from chest X-rays. " +
      "Be concise, neutral, and never invent patient information. Always remind the reader this is a screening tool, not a diagnosis.";

    const userPrompt = [
      `You are reviewing ${total} recent cardiomegaly screening results.`,
      `- Positive predictions: ${positive} / ${total}`,
      `- Average probability: ${(avgProb * 100).toFixed(1)}%`,
      "",
      "Scans (newest first):",
      ...lines,
      "",
      "Please respond in markdown with these sections:",
      "## Summary",
      "A short paragraph summarizing the batch (volume, positive rate, notable cases).",
      "## Notable cases",
      "Up to 5 bullets calling out high-probability scans or scans worth a second look. Reference patients by name or ID if available.",
      "## Suggested next steps",
      "3-5 actionable bullets for the clinician (e.g. follow-up imaging, prioritization, data-quality issues).",
      "## Improvements for future scans",
      "2-4 bullets on how the user could improve dataset quality or workflow (image quality, missing patient info, etc.).",
    ].join("\n");

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit reached, please try again in a moment." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } },
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({
            error: "AI credits exhausted. Add credits in Settings → Workspace → Usage.",
          }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } },
        );
      }
      const text = await response.text();
      console.error("AI gateway error:", response.status, text);
      return new Response(JSON.stringify({ error: "AI gateway error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const data = await response.json();
    const content: string = data?.choices?.[0]?.message?.content ?? "";

    return new Response(
      JSON.stringify({
        summary: content,
        stats: { total, positive, avg_probability: avgProb },
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  } catch (e) {
    console.error("summarize-history error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
