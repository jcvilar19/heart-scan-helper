// Per-scan AI insight: takes a single classification result and returns a
// short clinical-style summary + suggestions. Used on both the scanner page
// (live results) and the history page (saved scans).

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
    const { scan } = (await req.json()) as { scan: ScanInput };
    if (!scan || typeof scan.probability !== "number") {
      return new Response(JSON.stringify({ error: "Missing or invalid scan." }), {
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

    const verdict = scan.prediction === 1 ? "POSITIVE indication" : "negative";
    const probPct = (scan.probability * 100).toFixed(1);
    const who =
      [scan.patient_name, scan.patient_id ? `#${scan.patient_id}` : null]
        .filter(Boolean)
        .join(" ") ||
      scan.image_name ||
      "an unidentified patient";

    const systemPrompt =
      "You are a clinical AI assistant helping a clinician interpret a single cardiomegaly screening result from a chest X-ray. " +
      "Be concise, neutral, and never invent patient details that were not provided. Always remind the reader this is a screening tool, not a diagnosis. " +
      "Respond in markdown, in fewer than 180 words.";

    const userPrompt = [
      `Scan for ${who}.`,
      `- Pathology screened: ${scan.pathology ?? "cardiomegaly"}`,
      `- Model verdict: ${verdict}`,
      `- Probability: ${probPct}%`,
      scan.notes ? `- Clinician notes: ${scan.notes.slice(0, 400)}` : null,
      "",
      "Respond with these short sections:",
      "**Interpretation** — 1-2 sentences on what the result means in plain language.",
      "**Suggested next steps** — 2-3 bullets the clinician could take (e.g. follow-up imaging, second read).",
      "**Improvements** — 1-2 bullets on how to improve future scans for this patient (image quality, missing info, etc.).",
    ]
      .filter(Boolean)
      .join("\n");

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
    return new Response(JSON.stringify({ summary: content }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("summarize-scan error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
