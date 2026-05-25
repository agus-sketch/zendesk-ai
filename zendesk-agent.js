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
    try { res.status(r.status).json(JSON.parse(text)); }
    catch { res.status(r.status).send(text); }
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

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`\n✅ Zendesk AI running at http://localhost:${PORT}`);
  console.log(ZENDESK_SUBDOMAIN
    ? `   ✓ Instance: ${ZENDESK_SUBDOMAIN}.zendesk.com`
    : "   ⚠ ZENDESK_SUBDOMAIN not set — add it to .env");
  console.log("   Users authenticate with their own email + API token in the UI\n");
  ensureOllama().catch(() => {});
});

export default app;
