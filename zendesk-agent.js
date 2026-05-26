import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { config } from "dotenv";

config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const ZENDESK_SUBDOMAIN = process.env.ZENDESK_SUBDOMAIN || "";
const ZENDESK_BASE      = `https://${ZENDESK_SUBDOMAIN}.zendesk.com/api/v2`;
const SERVER_ZD_EMAIL   = process.env.ZENDESK_EMAIL  || "";
const SERVER_ZD_TOKEN   = process.env.ZENDESK_TOKEN  || "";

// Server-side LLM for the public reader (no key required from users)
const SERVER_GROQ_KEY      = process.env.GROQ_API_KEY      || "";
const SERVER_OPENAI_KEY    = process.env.OPENAI_API_KEY    || "";
const SERVER_ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY || "";
const SERVER_LLM_PROVIDER  = SERVER_GROQ_KEY ? "groq" : SERVER_OPENAI_KEY ? "openai" : SERVER_ANTHROPIC_KEY ? "anthropic" : "none";
const SERVER_LLM_KEY       = SERVER_GROQ_KEY || SERVER_OPENAI_KEY || SERVER_ANTHROPIC_KEY || "";

function zendeskAuth(email, token) {
  return `Basic ${Buffer.from(`${email}/token:${token}`).toString("base64")}`;
}

async function ensureOllama() {
  try {
    await fetch("http://localhost:11434");
    console.log("✓ Ollama already running");
  } catch {
    console.log("⚙ Starting Ollama...");
    spawn("ollama", ["serve"], { detached: true, stdio: "ignore" }).unref();
    for (let i = 0; i < 20; i++) {
      await new Promise(r => setTimeout(r, 500));
      try { await fetch("http://localhost:11434"); console.log("✓ Ollama ready"); return; } catch {}
    }
    console.log("⚠ Ollama didn't respond — continuing anyway");
  }
}

async function installedOllamaModels() {
  try {
    const r = await fetch("http://localhost:11434/api/tags");
    const d = await r.json();
    return (d.models || []).map(m => m.name);
  } catch { return []; }
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "4mb" }));
app.use(express.static(path.join(__dirname, "public")));

const LLM_URLS = {
  groq:      "https://api.groq.com/openai/v1/chat/completions",
  openai:    "https://api.openai.com/v1/chat/completions",
  ollama:    "http://localhost:11434/v1/chat/completions",
  anthropic: "https://api.anthropic.com/v1/messages",
};
const LLM_MODELS = {
  groq:      "llama-3.3-70b-versatile",
  openai:    "gpt-4o-mini",
  ollama:    "qwen2.5:14b",
  anthropic: "claude-haiku-4-5",
};

// ── Reader UI ─────────────────────────────────────────────────────────────────
app.get("/reader", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "reader.html"));
});

// ── Public read-only Zendesk proxy ────────────────────────────────────────────
// No user credentials required. Uses server-side creds if set (ZENDESK_EMAIL +
// ZENDESK_TOKEN in .env), otherwise falls back to unauthenticated access which
// works for any publicly visible Zendesk Help Center.
app.post("/zd-public", async (req, res) => {
  if (!ZENDESK_SUBDOMAIN) {
    return res.status(503).json({ error: "ZENDESK_SUBDOMAIN not configured." });
  }
  const { endpoint } = req.body || {};
  if (!endpoint || typeof endpoint !== "string") {
    return res.status(400).json({ error: "Missing endpoint" });
  }
  // Whitelist: only read-only Help Center paths
  const isRelative = !endpoint.startsWith("http");
  const isSafeRelative = isRelative && /^\/help_center\/(categories|sections|articles)(\/|\?|$)/.test(endpoint);
  const isSafeAbsolute = !isRelative && endpoint.startsWith(ZENDESK_BASE + "/help_center/");
  if (!isSafeRelative && !isSafeAbsolute) {
    return res.status(403).json({ error: "Not allowed" });
  }
  try {
    const url = isRelative ? ZENDESK_BASE + endpoint : endpoint;
    const headers = { "Content-Type": "application/json", Accept: "application/json" };
    // Use server credentials if configured, otherwise try unauthenticated (public HC)
    if (SERVER_ZD_EMAIL && SERVER_ZD_TOKEN) {
      headers.Authorization = zendeskAuth(SERVER_ZD_EMAIL, SERVER_ZD_TOKEN);
    }
    const r = await fetch(url, { headers });
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); }
    catch { data = { error: `Zendesk error (${r.status})` }; }
    res.status(r.status).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Server-side LLM helper ────────────────────────────────────────────────────
async function serverLLM(messages) {
  if (SERVER_LLM_PROVIDER === "none") throw new Error("AI not configured on the server. Add GROQ_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) to your .env file.");

  if (SERVER_LLM_PROVIDER === "anthropic") {
    const sys  = messages.filter(m => m.role === "system").map(m => m.content).join("\n\n");
    const conv = messages.filter(m => m.role !== "system");
    const r = await fetch(LLM_URLS.anthropic, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": SERVER_LLM_KEY, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: LLM_MODELS.anthropic, messages: conv, max_tokens: 1024, temperature: 0.2, ...(sys ? { system: sys } : {}) }),
    });
    const d = await r.json();
    return (d.content || []).map(b => b.text || "").join("");
  }

  const r = await fetch(LLM_URLS[SERVER_LLM_PROVIDER] || LLM_URLS.groq, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${SERVER_LLM_KEY}` },
    body: JSON.stringify({
      model: LLM_MODELS[SERVER_LLM_PROVIDER] || LLM_MODELS.groq,
      messages, temperature: 0.2, max_tokens: 1024,
      response_format: { type: "json_object" },
    }),
  });
  const d = await r.json();
  return d.choices?.[0]?.message?.content || "{}";
}

// ── Public AI reader chat ─────────────────────────────────────────────────────
app.post("/reader-chat", async (req, res) => {
  if (!ZENDESK_SUBDOMAIN) return res.status(503).json({ error: "ZENDESK_SUBDOMAIN not configured." });
  if (SERVER_LLM_PROVIDER === "none") return res.status(503).json({ error: "AI not configured. Add GROQ_API_KEY to .env." });

  const { question } = req.body || {};
  if (!question?.trim()) return res.status(400).json({ error: "Missing question" });

  const zdHeaders = { Accept: "application/json" };
  if (SERVER_ZD_EMAIL && SERVER_ZD_TOKEN) zdHeaders.Authorization = zendeskAuth(SERVER_ZD_EMAIL, SERVER_ZD_TOKEN);

  try {
    // 1. Search Zendesk for relevant articles
    const searchRes = await fetch(
      `${ZENDESK_BASE}/help_center/articles/search?query=${encodeURIComponent(question)}&per_page=5`,
      { headers: zdHeaders }
    );
    const searchData = await searchRes.json();
    const topArticles = (searchData.results || []).filter(a => !a.draft).slice(0, 3);

    if (!topArticles.length) {
      return res.json({
        answer: "<p>I couldn't find articles related to your question. Try rephrasing it or use different keywords.</p>",
        sources: [],
      });
    }

    // 2. Fetch full article bodies (strip HTML, cap at 1500 chars each)
    const withContent = await Promise.all(topArticles.map(async a => {
      try {
        const r = await fetch(`${ZENDESK_BASE}/help_center/articles/${a.id}`, { headers: zdHeaders });
        const d = await r.json();
        const text = (d.article?.body || "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 1500);
        return { id: a.id, title: a.title, text };
      } catch { return { id: a.id, title: a.title, text: "" }; }
    }));

    // 3. Ask the LLM
    const context = withContent.map(a => `# ${a.title}\n${a.text}`).join("\n\n---\n\n");
    const raw = await serverLLM([
      { role: "system", content: `You are a friendly Help Center assistant. Answer the user's question using ONLY the provided article content. Be concise and helpful. Use HTML: <p>, <ul>, <li>, <strong>. If the articles don't fully answer the question, say so honestly — never invent information. Return only a valid JSON object with one key: {"answer":"<HTML string>"}` },
      { role: "user", content: `Articles:\n${context}\n\nQuestion: ${question}` },
    ]);

    let answer;
    try {
      const parsed = JSON.parse(raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim());
      answer = parsed.answer || raw;
    } catch { answer = `<p>${raw}</p>`; }

    res.json({ answer, sources: withContent.map(a => ({ id: a.id, title: a.title })) });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/config", (req, res) => {
  res.json({ subdomain: ZENDESK_SUBDOMAIN });
});

app.get("/ollama-models", async (req, res) => {
  try {
    const r = await fetch("http://localhost:11434/api/tags");
    const d = await r.json();
    res.json({ models: (d.models || []).map(m => m.name) });
  } catch {
    res.json({ models: [] });
  }
});

// ── Zendesk proxy ──────────────────────────────────────────────────────────────
app.post("/zendesk", async (req, res) => {
  const { method, endpoint, body, email, token } = req.body;
  if (!endpoint)        return res.status(400).json({ error: "Missing endpoint" });
  if (!email || !token) return res.status(400).json({ error: "Missing Zendesk credentials" });
  if (!ZENDESK_SUBDOMAIN) return res.status(400).json({ error: "ZENDESK_SUBDOMAIN not set in .env" });
  try {
    const opts = {
      method: method || "GET",
      headers: {
        Authorization: zendeskAuth(email, token),
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    };
    if (body && method !== "GET") opts.body = JSON.stringify(body);
    const url = endpoint.startsWith("http") ? endpoint : ZENDESK_BASE + endpoint;
    const r = await fetch(url, opts);
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); }
    catch { data = { error: `Zendesk error (${r.status}): ${text.slice(0, 300)}` }; }
    res.status(r.status).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── LLM proxy ─────────────────────────────────────────────────────────────────
app.post("/llm", async (req, res) => {
  const { messages, prompt, llmKey, provider = "groq" } = req.body;
  if (!llmKey && provider !== "ollama") return res.status(400).json({ error: "Missing LLM API key" });

  const url   = LLM_URLS[provider]  || LLM_URLS.groq;
  const model = req.body.model      || LLM_MODELS[provider] || LLM_MODELS.groq;
  const msgs  = messages || [{ role: "user", content: prompt }];

  if (provider === "anthropic") return handleAnthropic({ res, llmKey, url, model, msgs });

  const headers = { "Content-Type": "application/json" };
  if (llmKey) headers.Authorization = `Bearer ${llmKey}`;

  const payload = {
    model,
    messages: msgs,
    temperature: 0.2,
    max_tokens: 2048,
    response_format: { type: "json_object" },
  };
  if (provider === "ollama") payload.keep_alive = "30m";

  const ctrl  = new AbortController();
  const timeout = provider === "ollama" ? 180000 : 30000;
  const timer = setTimeout(() => ctrl.abort(), timeout);

  try {
    const r = await fetch(url, { method: "POST", headers, body: JSON.stringify(payload), signal: ctrl.signal });
    clearTimeout(timer);
    res.json(await r.json());
  } catch (e) {
    clearTimeout(timer);
    const msg = e.name === "AbortError" ? "LLM timed out — try again" : e.message;
    if (!res.headersSent) res.status(500).json({ error: msg });
  }
});

async function handleAnthropic({ res, llmKey, url, model, msgs }) {
  const sys  = msgs.filter(m => m.role === "system").map(m => typeof m.content === "string" ? m.content : JSON.stringify(m.content)).join("\n\n");
  const conv = msgs.filter(m => m.role !== "system");
  const payload = { model, messages: conv, temperature: 0.2, max_tokens: 2048 };
  if (sys) payload.system = sys;

  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60000);
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": llmKey, "anthropic-version": "2023-06-01" },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    if (!r.ok) { clearTimeout(timer); return res.status(r.status).json({ error: `Anthropic: ${await r.text()}` }); }
    clearTimeout(timer);
    const data = await r.json();
    const text = (data.content || []).map(b => b.text || "").join("");
    res.json({ choices: [{ message: { role: "assistant", content: text } }] });
  } catch (e) {
    clearTimeout(timer);
    const msg = e.name === "AbortError" ? "LLM timed out — try again" : e.message;
    if (!res.headersSent) res.status(500).json({ error: msg });
  }
}

// ── Warm Ollama model ─────────────────────────────────────────────────────────
app.post("/warm", async (req, res) => {
  const { model } = req.body || {};
  if (!model) return res.status(400).json({ error: "Missing model" });
  try {
    await fetch("http://localhost:11434/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }], max_tokens: 1, keep_alive: "30m" }),
    });
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const isMain = process.argv[1] === fileURLToPath(import.meta.url);
if (isMain) {
  const PORT = process.env.PORT || 3001;
  app.listen(PORT, () => {
    console.log(`\n✅ Zendesk AI running at http://localhost:${PORT}`);
    console.log(ZENDESK_SUBDOMAIN
      ? `   ✓ Instance: ${ZENDESK_SUBDOMAIN}.zendesk.com`
      : "   ⚠ ZENDESK_SUBDOMAIN not set — add it to .env");
    console.log("   Users authenticate with their own email + API token in the UI\n");
    ensureOllama().catch(() => {});
  });
}

export default app;
