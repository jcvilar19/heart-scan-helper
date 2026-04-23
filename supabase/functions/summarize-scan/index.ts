// Per-scan AI insight using Lovable AI Gateway
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

type Scan = {
  patient_name?: string | null;
  patient_id?: string | null;
  image_name?: string | null;
  probability?: number | null;
  prediction?: number | null;
  pathology?: string | null;
  notes?: string | null;
  created_at?: string | null;
};

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      return new Response(JSON.stringify({ error: "LOVABLE_API_KEY is not configured" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const { scan } = (await req.json()) as { scan: Scan };
    if (!scan) {
      return new Response(JSON.stringify({ error: "Missing 'scan' in request body." }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const probabilityPct =
      typeof scan.probability === "number" ? `${(scan.probability * 100).toFixed(1)}%` : "N/A";
    const verdict =
      scan.prediction === 1 ? "Potential indication" : scan.prediction === 0 ? "No clear indication" : "Unknown";

    const userContent = [
      `Patient name: ${scan.patient_name || "N/A"}`,
      `Patient ID: ${scan.patient_id || "N/A"}`,
      `Image: ${scan.image_name || "N/A"}`,
      `Pathology: ${scan.pathology || "cardiomegaly"}`,
      `Model verdict: ${verdict}`,
      `Probability: ${probabilityPct}`,
      `Clinical notes: ${scan.notes?.trim() || "(none)"}`,
    ].join("\n");

    const systemPrompt = [
      "You are an assistant supporting a radiologist reviewing a single chest X-ray classified by an AI model for cardiomegaly.",
      "Your output is decision support — never a diagnosis.",
      "Respond in concise markdown with these three sections (use ### headings):",
      "### Interpretation — 1-2 sentences placing the model verdict and confidence in clinical context.",
      "### Suggested next steps — 2-4 bullet points (clinical correlation, additional imaging, follow-up).",
      "### Improvements — 1-2 bullets on how the user can improve the input (e.g. add notes, retake image, verify patient info).",
      "Always remind the clinician that this is decision support, not a diagnosis.",
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
          { role: "user", content: userContent },
        ],
      }),
    });

    if (response.status === 429) {
      return new Response(
        JSON.stringify({ error: "Rate limit reached. Please try again in a moment." }),
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
    if (!response.ok) {
      const text = await response.text();
      console.error("AI gateway error:", response.status, text);
      return new Response(JSON.stringify({ error: "AI gateway error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const data = await response.json();
    const summary = data?.choices?.[0]?.message?.content as string | undefined;
    if (!summary) {
      return new Response(JSON.stringify({ error: "Empty response from AI." }), {
        status: 502,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify({ summary }), {
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
