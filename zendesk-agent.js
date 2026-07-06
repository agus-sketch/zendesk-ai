import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { config } from "dotenv";
import { waitUntil } from "@vercel/functions";

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

// ── Upstash KV helpers ────────────────────────────────────────────────────────
async function kvGet(key) {
  const r = await fetch(`${process.env.KV_REST_API_URL}/get/${encodeURIComponent(key)}`, {
    headers: { Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}` },
  });
  const d = await r.json();
  return d.result ?? null;
}
async function kvSet(key, value) {
  await fetch(`${process.env.KV_REST_API_URL}/set/${encodeURIComponent(key)}/${encodeURIComponent(value)}`, {
    headers: { Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}` },
  });
}

async function getZendeskTokenForUser(slackUserId) {
  const raw = await kvGet(`zendesk:${slackUserId}`);
  if (!raw) return null;
  try { return JSON.parse(raw).access_token; } catch { return raw; }
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
  anthropic: "claude-haiku-4-5-20251001",
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
async function serverLLM(messages, provider = SERVER_LLM_PROVIDER, key = SERVER_LLM_KEY, json = true) {
  if (provider === "anthropic") {
    const sys  = messages.filter(m => m.role === "system").map(m => m.content).join("\n\n");
    const conv = messages.filter(m => m.role !== "system");
    const apiMessages = json ? [...conv, { role: "assistant", content: "{" }] : conv;
    const r = await fetch(LLM_URLS.anthropic, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: LLM_MODELS.anthropic, messages: apiMessages, max_tokens: 1024, temperature: 0.2, ...(sys ? { system: sys } : {}) }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error.message || JSON.stringify(d.error));
    const text = (d.content || []).map(b => b.text || "").join("");
    return json ? "{" + text : text;
  }

  const r = await fetch(LLM_URLS[provider] || LLM_URLS.groq, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: LLM_MODELS[provider] || LLM_MODELS.groq,
      messages, temperature: 0.2, max_tokens: 1024,
      response_format: { type: "json_object" },
    }),
  });
  const d = await r.json();
  if (d.error) throw new Error(d.error.message || JSON.stringify(d.error));
  return d.choices?.[0]?.message?.content || "{}";
}

// ── Public AI reader chat ─────────────────────────────────────────────────────
app.post("/reader-chat", async (req, res) => {
  if (!ZENDESK_SUBDOMAIN) return res.status(503).json({ error: "ZENDESK_SUBDOMAIN not configured." });

  const { question, llmKey: clientKey, provider: clientProvider } = req.body || {};

  // Prefer client-supplied key; fall back to server-side
  const useProvider = clientProvider || SERVER_LLM_PROVIDER;
  const useKey      = clientKey      || SERVER_LLM_KEY;
  if (!useKey && useProvider !== "ollama") {
    return res.status(503).json({ error: "No API key available. Enter your key in the settings above or ask the admin to configure one on the server." });
  }
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
    ], useProvider, useKey);

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
  const wantsStream = !!req.body.stream;

  if (provider === "anthropic") return handleAnthropic({ res, llmKey, url, model, msgs, wantsStream });

  const headers = { "Content-Type": "application/json" };
  if (llmKey) headers.Authorization = `Bearer ${llmKey}`;

  const ctrl  = new AbortController();
  const timeout = provider === "ollama" ? 180000 : 30000;
  const timer = setTimeout(() => ctrl.abort(), timeout);

  if (wantsStream) {
    const payload = {
      model,
      messages: msgs,
      temperature: 0.2,
      max_tokens: 2048,
      stream: true,
    };
    if (provider === "ollama") payload.keep_alive = "30m";

    try {
      const r = await fetch(url, { method: "POST", headers, body: JSON.stringify(payload), signal: ctrl.signal });
      clearTimeout(timer);
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      if (!r.ok) { res.end(`data: ${JSON.stringify({ error: `LLM error ${r.status}` })}\n\n`); return; }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) { res.end(); return; }
          res.write(dec.decode(value, { stream: true }));
        }
      } catch { res.end(); }
      return;
    } catch (e) {
      clearTimeout(timer);
      const msg = e.name === "AbortError" ? "LLM timed out — try again" : e.message;
      if (!res.headersSent) res.status(500).json({ error: msg });
      return;
    }
  }

  const payload = {
    model,
    messages: msgs,
    temperature: 0.2,
    max_tokens: 2048,
    response_format: { type: "json_object" },
  };
  if (provider === "ollama") payload.keep_alive = "30m";

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

async function handleAnthropic({ res, llmKey, url, model, msgs, wantsStream }) {
  const sys  = msgs.filter(m => m.role === "system").map(m => typeof m.content === "string" ? m.content : JSON.stringify(m.content)).join("\n\n");
  const conv = msgs.filter(m => m.role !== "system");
  const convWithPrefill = [...conv, { role: "assistant", content: "{" }];
  const payload = { model, messages: convWithPrefill, temperature: 0.2, max_tokens: 2048 };
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
    const text = "{" + (data.content || []).map(b => b.text || "").join("");
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

// ── Zendesk OAuth routes ──────────────────────────────────────────────────────
const ZD_OAUTH_CLIENT_ID = process.env.ZENDESK_OAUTH_CLIENT_ID || "zendesk_ai_agent";

app.get("/auth/zendesk", (req, res) => {
  const { slack_user_id } = req.query;
  if (!slack_user_id) return res.status(400).send("Missing slack_user_id");
  const params = new URLSearchParams({
    response_type: "code",
    redirect_uri: "https://zendesk-ai.vercel.app/auth/zendesk/callback",
    client_id: ZD_OAUTH_CLIENT_ID,
    scope: "read write",
    state: slack_user_id,
  });
  res.redirect(`https://${ZENDESK_SUBDOMAIN}.zendesk.com/oauth/authorizations/new?${params}`);
});

app.get("/auth/zendesk/callback", async (req, res) => {
  console.log("Zendesk callback params:", JSON.stringify(req.query));
  const { code, state: slackUserId, error, error_description } = req.query;
  if (error) return res.status(400).send(`Zendesk OAuth error: ${error} — ${error_description || ""}`);
  if (!code || !slackUserId) return res.status(400).send(`Missing code or state. Got: ${JSON.stringify(req.query)}`);
  try {
    const r = await fetch(`https://${ZENDESK_SUBDOMAIN}.zendesk.com/oauth/tokens`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grant_type: "authorization_code",
        code,
        client_id: ZD_OAUTH_CLIENT_ID,
        client_secret: process.env.ZENDESK_OAUTH_CLIENT_SECRET,
        redirect_uri: "https://zendesk-ai.vercel.app/auth/zendesk/callback",
        scope: "read",
      }),
    });
    const d = await r.json();
    if (!d.access_token) return res.status(400).send(`OAuth error: ${JSON.stringify(d)}`);
    await kvSet(`zendesk:${slackUserId}`, JSON.stringify({ access_token: d.access_token }));
    res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Zendesk Connected</title>
  <style>
    body { margin: 0; display: flex; align-items: center; justify-content: center; min-height: 100vh; background: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .card { background: white; border-radius: 16px; padding: 48px 56px; text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,0.08); max-width: 400px; }
    .icon { font-size: 56px; margin-bottom: 16px; }
    h1 { margin: 0 0 8px; font-size: 22px; color: #111; }
    p { margin: 0 0 24px; color: #6b7280; font-size: 15px; line-height: 1.5; }
    .countdown { font-size: 13px; color: #9ca3af; }
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">✅</div>
    <h1>Zendesk Connected!</h1>
    <p>Your account is linked. You can now ask questions directly in Slack and get answers from your Zendesk Help Center.</p>
    <div class="countdown">This tab will close in <span id="n">3</span> seconds…</div>
  </div>
  <script>
    let s = 3;
    const el = document.getElementById("n");
    const t = setInterval(() => { el.textContent = --s; if (s <= 0) { clearInterval(t); window.close(); } }, 1000);
  </script>
</body>
</html>`);
  } catch (e) {
    res.status(500).send(`Error: ${e.message}`);
  }
});

// ── Zendesk agent tools (for Claude tool_use) ────────────────────────────────
const ZENDESK_TOOLS = [
  {
    name: "search_articles",
    description: "Search Help Center articles by keyword. Use this to find articles before reading or editing them.",
    input_schema: { type: "object", properties: { query: { type: "string", description: "Search query" } }, required: ["query"] },
  },
  {
    name: "get_article",
    description: "Get the full content of a specific article by ID.",
    input_schema: { type: "object", properties: { article_id: { type: "number", description: "The article ID" } }, required: ["article_id"] },
  },
  {
    name: "update_article",
    description: "Update an existing article's title and/or body HTML content.",
    input_schema: {
      type: "object",
      properties: {
        article_id: { type: "number", description: "The article ID to update" },
        title: { type: "string", description: "New title (omit to keep existing)" },
        body: { type: "string", description: "New HTML body (omit to keep existing)" },
      },
      required: ["article_id"],
    },
  },
  {
    name: "create_article",
    description: "Create a new article in a section.",
    input_schema: {
      type: "object",
      properties: {
        section_id: { type: "number", description: "Section ID where the article will live" },
        title: { type: "string" },
        body: { type: "string", description: "HTML body content" },
      },
      required: ["section_id", "title", "body"],
    },
  },
  {
    name: "list_sections",
    description: "List all Help Center sections with their IDs, names, and category IDs.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "create_section",
    description: "Create a new section inside a category.",
    input_schema: {
      type: "object",
      properties: {
        category_id: { type: "number", description: "Category ID" },
        name: { type: "string", description: "Section name" },
        description: { type: "string", description: "Optional description" },
      },
      required: ["category_id", "name"],
    },
  },
  {
    name: "list_categories",
    description: "List all Help Center categories with their IDs and names.",
    input_schema: { type: "object", properties: {} },
  },
];

async function runZendeskTool(name, input, zdHeaders) {
  switch (name) {
    case "search_articles": {
      const r = await fetch(`${ZENDESK_BASE}/help_center/articles/search?query=${encodeURIComponent(input.query)}&per_page=10`, { headers: zdHeaders });
      const d = await r.json();
      return (d.results || []).filter(a => !a.draft).slice(0, 5).map(a => ({ id: a.id, title: a.title, url: a.html_url, section_id: a.section_id }));
    }
    case "get_article": {
      const r = await fetch(`${ZENDESK_BASE}/help_center/articles/${input.article_id}`, { headers: zdHeaders });
      const d = await r.json();
      const a = d.article;
      if (!a) return { error: "Article not found" };
      return { id: a.id, title: a.title, body: a.body, section_id: a.section_id, url: a.html_url };
    }
    case "update_article": {
      const body = {};
      if (input.title) body.title = input.title;
      if (input.body) body.body = input.body;
      const r = await fetch(`${ZENDESK_BASE}/help_center/articles/${input.article_id}`, {
        method: "PUT",
        headers: { ...zdHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ article: body }),
      });
      const d = await r.json();
      return r.ok ? { success: true, id: d.article?.id, title: d.article?.title } : { error: d.error || d.description || "Update failed" };
    }
    case "create_article": {
      const r = await fetch(`${ZENDESK_BASE}/help_center/sections/${input.section_id}/articles`, {
        method: "POST",
        headers: { ...zdHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ article: { title: input.title, body: input.body, locale: "en-us" } }),
      });
      const d = await r.json();
      return r.ok ? { success: true, id: d.article?.id, title: d.article?.title, url: d.article?.html_url } : { error: d.error || d.description || "Create failed" };
    }
    case "list_sections": {
      const r = await fetch(`${ZENDESK_BASE}/help_center/sections?per_page=100`, { headers: zdHeaders });
      const d = await r.json();
      return (d.sections || []).map(s => ({ id: s.id, name: s.name, category_id: s.category_id }));
    }
    case "create_section": {
      const body = { name: input.name, locale: "en-us" };
      if (input.description) body.description = input.description;
      const r = await fetch(`${ZENDESK_BASE}/help_center/categories/${input.category_id}/sections`, {
        method: "POST",
        headers: { ...zdHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ section: body }),
      });
      const d = await r.json();
      return r.ok ? { success: true, id: d.section?.id, name: d.section?.name } : { error: d.error || d.description || "Create failed" };
    }
    case "list_categories": {
      const r = await fetch(`${ZENDESK_BASE}/help_center/categories?per_page=100`, { headers: zdHeaders });
      const d = await r.json();
      return (d.categories || []).map(c => ({ id: c.id, name: c.name }));
    }
    default:
      return { error: "Unknown tool" };
  }
}

async function runZendeskAgent(userMessage, history, zdHeaders) {
  const messages = [...history, { role: "user", content: userMessage }];
  const system = "You are a helpful BankingBridge Zendesk Help Center agent. You can search, read, create, and edit Help Center articles and sections using the available tools. For questions, search for relevant articles first and answer from their content. For write operations (edit/create), use the appropriate tools and confirm what you did. Be concise. Use Slack formatting: *bold* not **bold**, _italic_, bullet points with •. No HTML in your final text replies.";

  let loopMessages = [...messages];

  for (let i = 0; i < 10; i++) {
    const r = await fetch(LLM_URLS.anthropic, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": SERVER_ANTHROPIC_KEY, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: LLM_MODELS.anthropic, system, messages: loopMessages, tools: ZENDESK_TOOLS, max_tokens: 2048, temperature: 0.2 }),
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error.message || JSON.stringify(d.error));

    if (d.stop_reason === "end_turn") {
      return (d.content || []).filter(b => b.type === "text").map(b => b.text).join("\n").trim();
    }

    if (d.stop_reason === "tool_use") {
      loopMessages.push({ role: "assistant", content: d.content });
      const toolResults = await Promise.all(
        (d.content || []).filter(b => b.type === "tool_use").map(async tool => ({
          type: "tool_result",
          tool_use_id: tool.id,
          content: JSON.stringify(await runZendeskTool(tool.name, tool.input, zdHeaders)),
        }))
      );
      loopMessages.push({ role: "user", content: toolResults });
    } else {
      return (d.content || []).filter(b => b.type === "text").map(b => b.text).join("\n").trim();
    }
  }
  return "I wasn't able to complete that. Please try again.";
}

// ── Per-channel conversation history ─────────────────────────────────────────
const slackChannelHistory = new Map();
function getChannelHistory(channelId) {
  if (!slackChannelHistory.has(channelId)) slackChannelHistory.set(channelId, []);
  return slackChannelHistory.get(channelId);
}

// ── Slack Events API ──────────────────────────────────────────────────────────
app.post("/slack/events", async (req, res) => {
  const { type, challenge, event } = req.body;

  if (type === "url_verification") return res.json({ challenge });
  if (!event || event.bot_id || event.type !== "message") return res.sendStatus(200);

  res.sendStatus(200);

  const slackToken = process.env.SLACK_BOT_TOKEN;
  if (!slackToken) return;

  const postMessage = async (text) => {
    const r = await fetch("https://slack.com/api/chat.postMessage", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${slackToken}` },
      body: JSON.stringify({ channel: event.channel, text }),
    });
    const d = await r.json();
    if (!d.ok) console.error("Slack postMessage error:", d.error, "channel:", event.channel);
    return d;
  };

  waitUntil((async () => {
    try {
      const question = event.text?.trim();
      if (!question) return;

      const accessToken = await getZendeskTokenForUser(event.user);
      if (!accessToken) {
        const connectUrl = `https://zendesk-ai.vercel.app/auth/zendesk?slack_user_id=${event.user}`;
        await postMessage(`Hi! I need to connect to your Zendesk account first.\n\n<${connectUrl}|Click here to connect Zendesk> — it takes about 10 seconds.`);
        return;
      }

      const history = getChannelHistory(event.channel);
      const zdHeaders = { Accept: "application/json", Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" };

      const answer = await runZendeskAgent(question, history, zdHeaders);

      // Keep history as plain user/assistant pairs
      history.push({ role: "user", content: question });
      history.push({ role: "assistant", content: answer });
      if (history.length > 20) history.splice(0, history.length - 20);

      await postMessage(answer);
    } catch (e) {
      console.error("Slack handler error:", e.message);
      await postMessage("Something went wrong. Please try again.").catch(() => {});
    }
  })());
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
