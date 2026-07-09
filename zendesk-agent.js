import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { config } from "dotenv";
import { waitUntil } from "@vercel/functions";
import { callLLM, LLM_MODELS } from "./lib/llmClients.js";

config();

// ── Manual Slack app configuration (api.slack.com/apps) ──────────────────────
// These endpoints exist in this file but Slack won't call them until you wire
// them up in the Slack app's admin config:
//   1. Slash Commands → Create New Command
//        Command:      /zendesk
//        Request URL:  <deployed-base-url>/slack/commands
//   2. Interactivity & Shortcuts → toggle on
//        Request URL:  <deployed-base-url>/slack/interactions
// Both routes reuse SLACK_SIGNING_SECRET (already required for /slack/events)
// to verify Slack's request signature — no additional secret is needed.
// Remember to reinstall/save the app after adding these so the new scopes and
// Request URLs actually take effect.

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const ZENDESK_SUBDOMAIN = process.env.ZENDESK_SUBDOMAIN || "";
const ZENDESK_BASE      = `https://${ZENDESK_SUBDOMAIN}.zendesk.com/api/v2`;
const SERVER_ZD_EMAIL   = process.env.ZENDESK_EMAIL  || "";
const SERVER_ZD_TOKEN   = process.env.ZENDESK_TOKEN  || "";

// ── Stale-article digest cron (see GET /cron/stale-articles below) ──────────
// CRON_SECRET:          required. Vercel's cron-auth pattern — the route only
//                        runs when `Authorization: Bearer <CRON_SECRET>` is
//                        present, so unset it and the route fails closed (500).
// SLACK_NOTIFY_CHANNEL:  required to actually post. Slack channel id (e.g.
//                        C0123456789) the weekly digest is posted to. If
//                        unset, the route still runs but logs a warning and
//                        skips posting instead of throwing.
// STALE_ARTICLE_DAYS:    optional, default 180. Articles whose `updated_at`
//                        is older than this many days are flagged as stale.
const CRON_SECRET          = process.env.CRON_SECRET || "";
const SLACK_NOTIFY_CHANNEL = process.env.SLACK_NOTIFY_CHANNEL || "";
const STALE_ARTICLE_DAYS   = parseInt(process.env.STALE_ARTICLE_DAYS, 10) || 180;

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
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`kvGet failed (${r.status}): ${text.slice(0, 300)}`);
  }
  const d = await r.json();
  if (d.error) throw new Error(`kvGet error: ${d.error}`);
  return d.result ?? null;
}
// ttlSeconds: optional expiry (Upstash REST maps path segments to Redis command args, so
// `SET key value EX <ttl>` becomes `/set/<key>/<value>/EX/<ttl>`).
async function kvSet(key, value, ttlSeconds) {
  const segments = [`set`, encodeURIComponent(key), encodeURIComponent(value)];
  if (ttlSeconds) segments.push("EX", String(ttlSeconds));
  const r = await fetch(`${process.env.KV_REST_API_URL}/${segments.join("/")}`, {
    headers: { Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}` },
  });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`kvSet failed (${r.status}): ${text.slice(0, 300)}`);
  }
  const d = await r.json().catch(() => ({}));
  if (d.error) throw new Error(`kvSet error: ${d.error}`);
  return d;
}
async function kvDel(key) {
  const r = await fetch(`${process.env.KV_REST_API_URL}/del/${encodeURIComponent(key)}`, {
    headers: { Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}` },
  });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    console.error(`kvDel failed (${r.status}) for key ${key}: ${text.slice(0, 300)}`);
    return false;
  }
  return true;
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

// Stash the raw body bytes so /slack/events, /slack/commands, and
// /slack/interactions can verify Slack's HMAC signature, which is computed
// over the exact raw request body. Slack sends JSON for the Events API but
// application/x-www-form-urlencoded for slash commands and interactivity
// payloads, so this same function is wired into both body parsers below —
// whichever one actually matches the request's Content-Type is the one that
// runs it and populates req.rawBody.
function captureRawBody(req, res, buf) {
  req.rawBody = buf;
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "4mb", verify: captureRawBody }));
app.use(express.urlencoded({ extended: true, verify: captureRawBody }));
app.use(express.static(path.join(__dirname, "public")));

// ── Slack request signature verification ─────────────────────────────────────
// https://api.slack.com/authentication/verifying-requests-from-slack
const SLACK_SIGNING_SECRET = process.env.SLACK_SIGNING_SECRET || "";
function verifySlackSignature(req, res, next) {
  if (!SLACK_SIGNING_SECRET) {
    console.error("SLACK_SIGNING_SECRET not configured — rejecting Slack request");
    return res.sendStatus(401);
  }
  const timestamp = req.headers["x-slack-request-timestamp"];
  const signature  = req.headers["x-slack-signature"];
  if (!timestamp || !signature) return res.sendStatus(401);

  // Reject requests older than 5 minutes to prevent replay attacks
  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(now - Number(timestamp)) > 60 * 5) return res.sendStatus(401);

  const rawBody = req.rawBody ? req.rawBody.toString("utf8") : "";
  const base = `v0:${timestamp}:${rawBody}`;
  const expected = "v0=" + crypto.createHmac("sha256", SLACK_SIGNING_SECRET).update(base).digest("hex");

  const sigBuf = Buffer.from(signature);
  const expBuf = Buffer.from(expected);
  if (sigBuf.length !== expBuf.length || !crypto.timingSafeEqual(sigBuf, expBuf)) {
    return res.sendStatus(401);
  }
  next();
}

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

// ── Stale Help Center article digest ─────────────────────────────────────────
// Lists articles oldest-updated-first and stops as soon as it hits one that's
// no longer stale (or after STALE_SCAN_MAX_PAGES pages) — since the API sort
// is ascending on updated_at, every article after that point is fresher still,
// so there's no need to scan the entire Help Center on every run.
const STALE_SCAN_MAX_PAGES  = 5;
const STALE_SCAN_PER_PAGE   = 100;
const STALE_DIGEST_MAX_ITEMS = 20;

async function findStaleArticles(staleDays, maxPages = STALE_SCAN_MAX_PAGES, perPage = STALE_SCAN_PER_PAGE) {
  const cutoffMs = Date.now() - staleDays * 24 * 60 * 60 * 1000;
  const zdHeaders = { Accept: "application/json" };
  if (SERVER_ZD_EMAIL && SERVER_ZD_TOKEN) zdHeaders.Authorization = zendeskAuth(SERVER_ZD_EMAIL, SERVER_ZD_TOKEN);

  let url = `${ZENDESK_BASE}/help_center/articles.json?sort_by=updated_at&sort_order=asc&per_page=${perPage}`;
  let checked = 0;
  const stale = [];

  for (let page = 0; page < maxPages && url; page++) {
    const r = await fetch(url, { headers: zdHeaders });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`Zendesk articles fetch failed (${r.status}): ${text.slice(0, 300)}`);
    }
    const d = await r.json();
    const articles = d.articles || [];

    let hitFresh = false;
    for (const a of articles) {
      checked++;
      // Skip drafts/archived articles when Zendesk exposes those flags on
      // this instance (confirmed `draft` exists; `archived` may not — handle
      // its absence gracefully rather than assuming the field is there).
      if (a.draft || a.archived) continue;

      const updatedMs = new Date(a.updated_at).getTime();
      if (!Number.isFinite(updatedMs) || updatedMs >= cutoffMs) {
        hitFresh = true;
        break;
      }
      stale.push({
        id: a.id,
        title: a.title || `Article ${a.id}`,
        updatedAt: a.updated_at,
        url: a.html_url || `https://${ZENDESK_SUBDOMAIN}.zendesk.com/hc/en-us/articles/${a.id}`,
      });
    }
    if (hitFresh) break;
    url = d.next_page || null;
  }

  return { stale, checked };
}

function daysSince(isoString) {
  const ms = Date.now() - new Date(isoString).getTime();
  return Math.max(0, Math.floor(ms / (24 * 60 * 60 * 1000)));
}

function formatAge(days) {
  if (days >= 365) { const y = Math.floor(days / 365); return `${y} year${y === 1 ? "" : "s"} ago`; }
  if (days >= 30)  { const m = Math.floor(days / 30);  return `${m} month${m === 1 ? "" : "s"} ago`; }
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function buildStaleDigestText(stale, staleDays, maxItems = STALE_DIGEST_MAX_ITEMS) {
  const shown = stale.slice(0, maxItems);
  const remaining = stale.length - shown.length;
  const lines = shown.map(a => `• <${a.url}|${a.title}> — last updated ${formatAge(daysSince(a.updatedAt))}`);
  let text = `*📋 Stale Help Center articles* (not updated in ${staleDays}+ days, ${stale.length} found)\n\n${lines.join("\n")}`;
  if (remaining > 0) text += `\n\n_...and ${remaining} more not shown._`;
  return text;
}

// ── Cron: weekly stale-article digest ────────────────────────────────────────
// Wired up in vercel.json's `crons` array. Vercel signs cron requests with a
// Bearer token matching CRON_SECRET — see
// https://vercel.com/docs/cron-jobs/manage-cron-jobs#securing-cron-jobs
app.get("/cron/stale-articles", async (req, res) => {
  if (!CRON_SECRET) {
    console.error("CRON_SECRET not configured — refusing to run /cron/stale-articles");
    return res.status(500).json({ error: "CRON_SECRET not configured on the server." });
  }
  if (req.headers.authorization !== `Bearer ${CRON_SECRET}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  if (!ZENDESK_SUBDOMAIN) {
    return res.status(503).json({ error: "ZENDESK_SUBDOMAIN not configured." });
  }

  try {
    const { stale, checked } = await findStaleArticles(STALE_ARTICLE_DAYS);

    if (!stale.length) {
      return res.json({ checked, stale: 0, posted: false });
    }

    if (!SLACK_NOTIFY_CHANNEL) {
      console.warn("SLACK_NOTIFY_CHANNEL not configured — skipping stale-article Slack digest");
      return res.json({ checked, stale: stale.length, posted: false, warning: "SLACK_NOTIFY_CHANNEL not configured." });
    }
    const slackToken = process.env.SLACK_BOT_TOKEN;
    if (!slackToken) {
      console.warn("SLACK_BOT_TOKEN not configured — skipping stale-article Slack digest");
      return res.json({ checked, stale: stale.length, posted: false, warning: "SLACK_BOT_TOKEN not configured." });
    }

    const text = buildStaleDigestText(stale, STALE_ARTICLE_DAYS);
    const r = await fetch("https://slack.com/api/chat.postMessage", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${slackToken}` },
      body: JSON.stringify({ channel: SLACK_NOTIFY_CHANNEL, text }),
    });
    const d = await r.json();
    if (!d.ok) console.error("Slack postMessage error (stale-articles digest):", d.error);

    res.json({ checked, stale: stale.length, posted: !!d.ok });
  } catch (e) {
    console.error("/cron/stale-articles error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

// ── Server-side LLM helper ────────────────────────────────────────────────────
async function serverLLM(messages, provider = SERVER_LLM_PROVIDER, key = SERVER_LLM_KEY, json = true) {
  const result = await callLLM({
    provider,
    model: LLM_MODELS[provider] || LLM_MODELS.groq,
    messages,
    apiKey: key,
    temperature: 0.2,
    maxTokens: 1024,
    jsonMode: json,
  });
  if (result.raw?.error) throw new Error(result.raw.error.message || JSON.stringify(result.raw.error));
  return result.text || (json ? "{}" : "");
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

// ── Admin web UI session persistence ─────────────────────────────────────────
// Chat/UI state only (chat history, last-viewed article, last search query) —
// deliberately separate from the `zendesk:<slackUserId>` OAuth token storage
// above, which holds actual Zendesk credentials. Keyed by an opaque id the
// client generates once and keeps in localStorage (`zendesk_session_id`), not
// by any Slack/Zendesk identity.
const SESSION_HISTORY_LIMIT = 30;

// Truncate any array-shaped chat/message history in the session payload
// before persisting, so a long-running admin session can't grow the KV value
// unbounded.
function capSessionData(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) return data;
  const capped = { ...data };
  for (const key of ["chatHistory", "messages", "history"]) {
    if (Array.isArray(capped[key]) && capped[key].length > SESSION_HISTORY_LIMIT) {
      capped[key] = capped[key].slice(-SESSION_HISTORY_LIMIT);
    }
  }
  return capped;
}

app.post("/session", async (req, res) => {
  const { id: bodyId, data } = req.body || {};
  if (data === undefined) return res.status(400).json({ error: "Missing data" });
  const id = bodyId || crypto.randomUUID();
  try {
    await kvSet(`web_session:${id}`, JSON.stringify(capSessionData(data)));
    res.json({ id });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/session/:id", async (req, res) => {
  try {
    const raw = await kvGet(`web_session:${req.params.id}`);
    if (!raw) return res.status(404).json({ error: "Not found" });
    let data;
    try { data = JSON.parse(raw); } catch { data = raw; }
    res.json({ data });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Bulk Help Center article flagging ────────────────────────────────────────
// Minimal, deliberately dumb persisted marker for the admin UI's "Flag as
// stale" bulk action — just records article ids in KV. No workflow (no
// notifications, no automatic un-flagging) is built on top of this yet.
app.post("/articles/flag", async (req, res) => {
  const { ids } = req.body || {};
  if (!Array.isArray(ids) || !ids.length) return res.status(400).json({ error: "Missing ids" });
  try {
    const raw = await kvGet("flagged_articles");
    let existing;
    try { existing = raw ? JSON.parse(raw) : []; } catch { existing = []; }
    if (!Array.isArray(existing)) existing = [];
    const merged = Array.from(new Set([...existing, ...ids.map(Number)].filter(Number.isFinite)));
    await kvSet("flagged_articles", JSON.stringify(merged));
    res.json({ ok: true, flagged: merged });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Zendesk proxy ──────────────────────────────────────────────────────────────
app.post("/zendesk", async (req, res) => {
  const { method, endpoint, body, email, token } = req.body;
  if (!endpoint)        return res.status(400).json({ error: "Missing endpoint" });
  // Only allow relative paths under ZENDESK_BASE — reject absolute/external URLs
  // and protocol-relative paths to prevent SSRF via the stored Zendesk auth header.
  if (typeof endpoint !== "string" || !endpoint.startsWith("/") || endpoint.startsWith("//") || endpoint.includes("://")) {
    return res.status(400).json({ error: "Invalid endpoint — must be a relative Zendesk API path" });
  }
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
    const url = ZENDESK_BASE + endpoint;
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
  const { messages, prompt, provider = "groq" } = req.body;
  const llmKey = req.body.llmKey || (provider === "anthropic" ? SERVER_ANTHROPIC_KEY : provider === "openai" ? SERVER_OPENAI_KEY : provider === "groq" ? SERVER_GROQ_KEY : "");
  if (!llmKey && provider !== "ollama") return res.status(400).json({ error: "Missing LLM API key" });

  const model = req.body.model || LLM_MODELS[provider] || LLM_MODELS.groq;
  const msgs  = messages || [{ role: "user", content: prompt }];
  const wantsStream = !!req.body.stream;

  if (provider === "anthropic") return handleAnthropic({ res, llmKey, model, msgs });

  const timeout = provider === "ollama" ? 180000 : 30000;

  if (wantsStream) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    try {
      const result = await callLLM({
        provider, model, messages: msgs, apiKey: llmKey,
        temperature: 0.2, maxTokens: 2048, stream: true,
        keepAlive: provider === "ollama", timeoutMs: timeout,
        onPartial: chunk => res.write(chunk),
      });
      if (!result.ok) { res.end(`data: ${JSON.stringify({ error: `LLM error ${result.status}` })}\n\n`); return; }
      res.end();
    } catch (e) {
      const msg = e.name === "AbortError" ? "LLM timed out — try again" : e.message;
      if (!res.headersSent) res.status(500).json({ error: msg });
    }
    return;
  }

  try {
    const result = await callLLM({
      provider, model, messages: msgs, apiKey: llmKey,
      temperature: 0.2, maxTokens: 2048, jsonMode: true,
      keepAlive: provider === "ollama", timeoutMs: timeout,
    });
    res.json(result.raw);
  } catch (e) {
    const msg = e.name === "AbortError" ? "LLM timed out — try again" : e.message;
    if (!res.headersSent) res.status(500).json({ error: msg });
  }
});

async function handleAnthropic({ res, llmKey, model, msgs }) {
  try {
    const result = await callLLM({
      provider: "anthropic", model, messages: msgs, apiKey: llmKey,
      temperature: 0.2, maxTokens: 2048, jsonMode: true, timeoutMs: 60000,
    });
    if (!result.ok) return res.status(result.status).json({ error: `Anthropic: ${result.rawText}` });
    res.json({ choices: [{ message: { role: "assistant", content: result.text } }] });
  } catch (e) {
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

// Slack tools include write operations (update_article, create_article, create_section),
// so the scope requested here must match what's exchanged in the callback below.
const ZD_OAUTH_SCOPE = "read write";
const OAUTH_STATE_TTL_SECONDS = 600; // 10 minutes

app.get("/auth/zendesk", async (req, res) => {
  const { slack_user_id } = req.query;
  if (!slack_user_id) return res.status(400).send("Missing slack_user_id");

  const state = crypto.randomBytes(16).toString("hex");
  try {
    await kvSet(`oauth_state:${state}`, slack_user_id, OAUTH_STATE_TTL_SECONDS);
  } catch (e) {
    console.error("Failed to store OAuth state:", e.message);
    return res.status(500).send("Failed to start the Zendesk connection flow. Please try again.");
  }

  const params = new URLSearchParams({
    response_type: "code",
    redirect_uri: "https://zendesk-ai.vercel.app/auth/zendesk/callback",
    client_id: ZD_OAUTH_CLIENT_ID,
    scope: ZD_OAUTH_SCOPE,
    state,
  });
  res.redirect(`https://${ZENDESK_SUBDOMAIN}.zendesk.com/oauth/authorizations/new?${params}`);
});

app.get("/auth/zendesk/callback", async (req, res) => {
  console.log("Zendesk callback params:", JSON.stringify(req.query));
  const { code, state, error, error_description } = req.query;
  if (error) return res.status(400).send(`Zendesk OAuth error: ${error} — ${error_description || ""}`);
  if (!code || !state) return res.status(400).send(`Missing code or state. Got: ${JSON.stringify(req.query)}`);

  let slackUserId;
  try {
    slackUserId = await kvGet(`oauth_state:${state}`);
  } catch (e) {
    console.error("Failed to look up OAuth state:", e.message);
    return res.status(500).send("Failed to verify the connection request. Please try again.");
  }
  if (!slackUserId) {
    return res.status(400).send("This connection link has expired or is invalid. Please restart from Slack.");
  }

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
        scope: ZD_OAUTH_SCOPE,
      }),
    });
    const d = await r.json();
    if (!d.access_token) return res.status(400).send(`OAuth error: ${JSON.stringify(d)}`);
    await kvSet(`zendesk:${slackUserId}`, JSON.stringify({ access_token: d.access_token }));
    await kvDel(`oauth_state:${state}`);
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

// Returns { text, articles }. `articles` collects any concrete Help Center
// articles (id/title/url) surfaced by search_articles or get_article tool
// calls during the loop, so callers (Slack routes) can attach "open article" /
// "suggest improvement" buttons only when there's something concrete to link.
function collectArticlesFromToolOutput(toolName, output, into) {
  if (toolName !== "search_articles" && toolName !== "get_article") return;
  const candidates = Array.isArray(output) ? output : [output];
  for (const a of candidates) {
    if (a && a.id != null && a.url && !a.error) into.set(a.id, { id: a.id, title: a.title || `Article ${a.id}`, url: a.url });
  }
}

async function runZendeskAgent(userMessage, history, zdHeaders) {
  const messages = [...history, { role: "user", content: userMessage }];
  const system = "You are a helpful BankingBridge Zendesk Help Center agent. You can search, read, create, and edit Help Center articles and sections using the available tools. For questions, search for relevant articles first and answer from their content. For write operations (edit/create), use the appropriate tools and confirm what you did. Be concise. Use Slack formatting: *bold* not **bold**, _italic_, bullet points with •. No HTML in your final text replies.";

  let loopMessages = [...messages];
  const foundArticles = new Map();

  for (let i = 0; i < 10; i++) {
    const result = await callLLM({
      provider: "anthropic",
      model: LLM_MODELS.anthropic,
      system,
      messages: loopMessages,
      tools: ZENDESK_TOOLS,
      maxTokens: 2048,
      temperature: 0.2,
      apiKey: SERVER_ANTHROPIC_KEY,
    });
    const d = result.raw || {};
    if (d.error) throw new Error(d.error.message || JSON.stringify(d.error));

    if (d.stop_reason === "end_turn") {
      const text = (d.content || []).filter(b => b.type === "text").map(b => b.text).join("\n").trim();
      return { text, articles: [...foundArticles.values()] };
    }

    if (d.stop_reason === "tool_use") {
      loopMessages.push({ role: "assistant", content: d.content });
      const toolResults = await Promise.all(
        (d.content || []).filter(b => b.type === "tool_use").map(async tool => {
          const output = await runZendeskTool(tool.name, tool.input, zdHeaders);
          collectArticlesFromToolOutput(tool.name, output, foundArticles);
          return { type: "tool_result", tool_use_id: tool.id, content: JSON.stringify(output) };
        })
      );
      loopMessages.push({ role: "user", content: toolResults });
    } else {
      const text = (d.content || []).filter(b => b.type === "text").map(b => b.text).join("\n").trim();
      return { text, articles: [...foundArticles.values()] };
    }
  }
  return { text: "I wasn't able to complete that. Please try again.", articles: [...foundArticles.values()] };
}

// ── Slack Block Kit helpers ───────────────────────────────────────────────────
// Encodes which article a "Suggest improvement" button refers to directly in
// the button's `value` — no server-side state needed to resolve the click.
function encodeSuggestImprovementValue(articleId) {
  return JSON.stringify({ action: "suggest_improvement", articleId });
}

// Only attach buttons when we have concrete articles with known ids/urls —
// plain conversational replies (or write confirmations with no articles
// looked up) get no blocks and just render as normal text.
function buildArticleBlocks(articles) {
  if (!Array.isArray(articles) || !articles.length) return null;
  const blocks = [];
  for (const a of articles.slice(0, 5)) {
    blocks.push({
      type: "section",
      text: { type: "mrkdwn", text: `*<${a.url}|${a.title}>*` },
    });
    blocks.push({
      type: "actions",
      block_id: `article_actions_${a.id}`,
      elements: [
        {
          type: "button",
          text: { type: "plain_text", text: "🔗 Open in Help Center", emoji: true },
          url: a.url,
        },
        {
          type: "button",
          text: { type: "plain_text", text: "✨ Suggest improvement", emoji: true },
          action_id: "suggest_improvement",
          value: encodeSuggestImprovementValue(a.id),
        },
      ],
    });
  }
  return blocks;
}

async function postToResponseUrl(responseUrl, payload) {
  try {
    const r = await fetch(responseUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) console.error("response_url post failed:", r.status, await r.text().catch(() => ""));
    return r;
  } catch (e) {
    console.error("response_url post error:", e.message);
    return null;
  }
}

// ── Per-channel conversation history ─────────────────────────────────────────
// Persisted in the same KV store used for OAuth tokens so history survives
// serverless cold starts (an in-memory Map does not survive across invocations
// on Vercel). Capped to the last CHANNEL_HISTORY_LIMIT messages before writing.
const CHANNEL_HISTORY_LIMIT = 20; // ~10 user/assistant turns
async function getChannelHistory(channelId) {
  try {
    const raw = await kvGet(`slack_history:${channelId}`);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.error("Failed to load channel history:", e.message);
    return [];
  }
}
async function saveChannelHistory(channelId, history) {
  const capped = history.slice(-CHANNEL_HISTORY_LIMIT);
  try {
    await kvSet(`slack_history:${channelId}`, JSON.stringify(capped));
  } catch (e) {
    console.error("Failed to save channel history:", e.message);
  }
}

// ── Slack Events API ──────────────────────────────────────────────────────────
app.post("/slack/events", verifySlackSignature, async (req, res) => {
  const { type, challenge, event } = req.body;

  if (type === "url_verification") return res.json({ challenge });
  if (!event || event.bot_id || event.type !== "message") return res.sendStatus(200);

  res.sendStatus(200);

  const slackToken = process.env.SLACK_BOT_TOKEN;
  if (!slackToken) return;

  const postMessage = async (text, articles = []) => {
    const body = { channel: event.channel, text };
    const blocks = buildArticleBlocks(articles);
    if (blocks) body.blocks = blocks;
    const r = await fetch("https://slack.com/api/chat.postMessage", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${slackToken}` },
      body: JSON.stringify(body),
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

      const history = await getChannelHistory(event.channel);
      const zdHeaders = { Accept: "application/json", Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" };

      const { text: answer, articles } = await runZendeskAgent(question, history, zdHeaders);

      // Keep history as plain user/assistant pairs, capped before persisting
      const updatedHistory = [...history, { role: "user", content: question }, { role: "assistant", content: answer }];
      await saveChannelHistory(event.channel, updatedHistory);

      await postMessage(answer, articles);
    } catch (e) {
      console.error("Slack handler error:", e.message);
      await postMessage("Something went wrong. Please try again.").catch(() => {});
    }
  })());
});

// ── Slack slash command: /zendesk ────────────────────────────────────────────
// Requires the "Slash Commands" feature configured in the Slack app (see the
// config comment near the top of this file). Slack requires an ack within 3
// seconds, so we respond immediately and do the real work in the background,
// delivering the final answer via response_url.
app.post("/slack/commands", verifySlackSignature, async (req, res) => {
  const { text, user_id: slackUserId, channel_id: channelId, response_url: responseUrl } = req.body;

  res.status(200).json({ response_type: "ephemeral", text: "Searching…" });

  if (!responseUrl) return; // nothing we can do without it

  waitUntil((async () => {
    try {
      const query = (text || "").trim() || "list recently updated articles";

      // Prefer the invoking user's own OAuth token (matches /slack/events'
      // write-capable flow); fall back to server-side read-only creds so
      // read-only searches still work for users who haven't connected yet.
      let zdHeaders;
      const accessToken = await getZendeskTokenForUser(slackUserId);
      if (accessToken) {
        zdHeaders = { Accept: "application/json", Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" };
      } else if (SERVER_ZD_EMAIL && SERVER_ZD_TOKEN) {
        zdHeaders = { Accept: "application/json", Authorization: zendeskAuth(SERVER_ZD_EMAIL, SERVER_ZD_TOKEN), "Content-Type": "application/json" };
      } else {
        const connectUrl = `https://zendesk-ai.vercel.app/auth/zendesk?slack_user_id=${slackUserId}`;
        await postToResponseUrl(responseUrl, {
          response_type: "ephemeral",
          text: `Hi! I need to connect to your Zendesk account first.\n\n<${connectUrl}|Click here to connect Zendesk> — it takes about 10 seconds.`,
        });
        return;
      }

      const history = await getChannelHistory(channelId);
      const { text: answer, articles } = await runZendeskAgent(query, history, zdHeaders);

      const updatedHistory = [...history, { role: "user", content: query }, { role: "assistant", content: answer }];
      await saveChannelHistory(channelId, updatedHistory);

      const payload = { response_type: "in_channel", text: answer };
      const blocks = buildArticleBlocks(articles);
      if (blocks) payload.blocks = blocks;
      await postToResponseUrl(responseUrl, payload);
    } catch (e) {
      console.error("/slack/commands handler error:", e.message);
      await postToResponseUrl(responseUrl, { response_type: "ephemeral", text: "Something went wrong. Please try again." }).catch(() => {});
    }
  })());
});

// ── Slack interactivity: button clicks ───────────────────────────────────────
// Requires "Interactivity & Shortcuts" enabled in the Slack app (see the
// config comment near the top of this file). Slack sends these as
// application/x-www-form-urlencoded with a single `payload` field holding a
// JSON string. Ack within 3 seconds, then do the real work in the background.
app.post("/slack/interactions", verifySlackSignature, async (req, res) => {
  res.status(200).send();

  let payload;
  try {
    payload = JSON.parse(req.body?.payload || "");
  } catch (e) {
    console.error("Failed to parse /slack/interactions payload:", e.message);
    return;
  }

  if (payload.type !== "block_actions") return;

  const action = (payload.actions || [])[0];
  const responseUrl = payload.response_url;
  if (!action || !responseUrl) return;

  // Only the "Suggest improvement" button carries a value we act on here —
  // the "Open in Help Center" button is a plain `url` button Slack handles
  // client-side, so no block_actions event is even sent for it.
  if (action.action_id !== "suggest_improvement" || !action.value) return;

  let decoded;
  try { decoded = JSON.parse(action.value); } catch { decoded = null; }
  const articleId = decoded?.action === "suggest_improvement" ? Number(decoded.articleId) : null;
  if (!articleId) {
    await postToResponseUrl(responseUrl, { replace_original: false, text: "Couldn't figure out which article this button refers to." });
    return;
  }

  waitUntil((async () => {
    try {
      if (!ZENDESK_SUBDOMAIN) {
        await postToResponseUrl(responseUrl, { replace_original: false, text: "ZENDESK_SUBDOMAIN isn't configured on the server." });
        return;
      }
      if (!SERVER_ANTHROPIC_KEY) {
        await postToResponseUrl(responseUrl, { replace_original: false, text: "No Anthropic API key configured on the server, so I can't generate suggestions." });
        return;
      }

      // Reuse the server-level read path (same creds as the public reader /
      // /zd-public) — improvement suggestions only need to read the article,
      // not the clicking user's own write-scoped OAuth token.
      const zdHeaders = { Accept: "application/json", "Content-Type": "application/json" };
      if (SERVER_ZD_EMAIL && SERVER_ZD_TOKEN) zdHeaders.Authorization = zendeskAuth(SERVER_ZD_EMAIL, SERVER_ZD_TOKEN);

      const article = await runZendeskTool("get_article", { article_id: articleId }, zdHeaders);
      if (!article || article.error || !article.body) {
        await postToResponseUrl(responseUrl, {
          replace_original: false,
          text: `Couldn't fetch article ${articleId}: ${article?.error || "not found"}`,
        });
        return;
      }

      const plainBody = article.body.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 6000);
      const result = await callLLM({
        provider: "anthropic",
        model: LLM_MODELS.anthropic,
        system: "You are a helpful BankingBridge Zendesk Help Center editor. Review the given Help Center article and suggest concrete improvements: clarity issues, missing or unclear steps, outdated information, and anything a reader might find confusing. Be specific and actionable — reference the relevant part of the article for each suggestion. Use Slack formatting: *bold* not **bold**, _italic_, bullet points with •. No HTML in your reply. Keep it concise.",
        messages: [{ role: "user", content: `Article title: ${article.title}\n\nArticle content:\n${plainBody}` }],
        maxTokens: 1024,
        temperature: 0.3,
        apiKey: SERVER_ANTHROPIC_KEY,
      });
      const d = result.raw || {};
      if (d.error) throw new Error(d.error.message || JSON.stringify(d.error));
      const suggestions = (d.content || []).filter(b => b.type === "text").map(b => b.text).join("\n").trim()
        || "I couldn't come up with suggestions for this article.";

      await postToResponseUrl(responseUrl, {
        replace_original: false,
        text: `*✨ Suggestions for <${article.url}|${article.title}>*\n\n${suggestions}`,
      });
    } catch (e) {
      console.error("suggest_improvement handler error:", e.message);
      await postToResponseUrl(responseUrl, { replace_original: false, text: "Something went wrong while generating suggestions. Please try again." }).catch(() => {});
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
