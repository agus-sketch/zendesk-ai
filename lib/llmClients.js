// Unified LLM provider dispatch, extracted from zendesk-agent.js to eliminate
// duplicated fetch/parsing logic across the /llm route, the reader-chat helper,
// and the Slack agent's Claude tool-use loop.
//
// Providers:
//   - groq / openai / ollama share the OpenAI-compatible chat-completions shape.
//   - anthropic uses its own Messages API shape (system as a top-level field,
//     `tools`/`tool_choice` pass-through, and the "{" assistant-prefill trick
//     for forcing valid JSON output when tools aren't in play).
//
// Anthropic is ALWAYS called non-streaming here, by design (see the /llm route's
// recent fix) — `stream` is accepted for API symmetry but ignored for anthropic.
import fetch from "node-fetch";

export const LLM_URLS = {
  groq:      "https://api.groq.com/openai/v1/chat/completions",
  openai:    "https://api.openai.com/v1/chat/completions",
  ollama:    "http://localhost:11434/v1/chat/completions",
  anthropic: "https://api.anthropic.com/v1/messages",
};

export const LLM_MODELS = {
  groq:      "llama-3.3-70b-versatile",
  openai:    "gpt-4o-mini",
  ollama:    "qwen2.5:14b",
  anthropic: "claude-haiku-4-5-20251001",
};

const ANTHROPIC_VERSION = "2023-06-01";

function buildAbort({ signal, timeoutMs }) {
  if (signal) return { signal, timer: null };
  if (!timeoutMs) return { signal: undefined, timer: null };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  return { signal: ctrl.signal, timer };
}

// ── Anthropic Messages API ────────────────────────────────────────────────────
async function callAnthropic({
  model, messages = [], system, apiKey, temperature, maxTokens,
  jsonMode, tools, toolChoice, timeoutMs, signal,
}) {
  // System prompt may arrive either as an explicit `system` param (the Slack
  // tool-use loop's style) or embedded as role:"system" messages (the
  // serverLLM / /llm route style) — support both and merge if both are present.
  const systemFromMessages = messages
    .filter(m => m.role === "system")
    .map(m => (typeof m.content === "string" ? m.content : JSON.stringify(m.content)))
    .join("\n\n");
  const conv = messages.filter(m => m.role !== "system");
  const sys = [system, systemFromMessages].filter(Boolean).join("\n\n");

  const hasTools = Array.isArray(tools) && tools.length > 0;
  // Never apply the JSON-prefill trick when tools are in play — an assistant
  // "{" prefill is incompatible with tool_use turns.
  const usePrefill = jsonMode && !hasTools;
  const apiMessages = usePrefill ? [...conv, { role: "assistant", content: "{" }] : conv;

  const payload = { model, messages: apiMessages, max_tokens: maxTokens, temperature };
  if (sys) payload.system = sys;
  if (hasTools) {
    payload.tools = tools;
    if (toolChoice) payload.tool_choice = toolChoice;
  }

  const { signal: effSignal, timer } = buildAbort({ signal, timeoutMs });
  try {
    const r = await fetch(LLM_URLS.anthropic, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": ANTHROPIC_VERSION,
      },
      body: JSON.stringify(payload),
      signal: effSignal,
    });
    if (timer) clearTimeout(timer);

    const rawText = await r.text();
    let raw = null;
    try { raw = JSON.parse(rawText); } catch { /* leave raw null, rawText still available */ }

    const content = raw?.content || [];
    const toolUses = hasTools ? content.filter(b => b.type === "tool_use") : undefined;
    const joined = content.map(b => b.text || "").join("");
    const text = usePrefill ? "{" + joined : joined;

    return {
      ok: r.ok,
      status: r.status,
      raw,
      rawText,
      text,
      toolUses,
      stopReason: raw?.stop_reason,
    };
  } catch (e) {
    if (timer) clearTimeout(timer);
    throw e;
  }
}

// ── OpenAI-compatible chat completions (groq / openai / ollama) ─────────────
async function callOpenAiCompatible({
  provider, model, messages, apiKey, temperature, maxTokens,
  jsonMode, stream, keepAlive, timeoutMs, signal, onPartial,
}) {
  const url = LLM_URLS[provider] || LLM_URLS.groq;

  const payload = {
    model,
    messages,
    temperature,
    max_tokens: maxTokens,
  };
  if (jsonMode) payload.response_format = { type: "json_object" };
  if (stream) payload.stream = true;
  if (keepAlive && provider === "ollama") payload.keep_alive = "30m";

  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  const { signal: effSignal, timer } = buildAbort({ signal, timeoutMs });
  try {
    const r = await fetch(url, { method: "POST", headers, body: JSON.stringify(payload), signal: effSignal });
    if (timer) clearTimeout(timer);

    if (stream) {
      if (!r.ok) return { ok: false, status: r.status };
      // node-fetch v3's Response.body is a Node.js Readable (not a WHATWG
      // ReadableStream), so it doesn't expose .getReader()/.read() — the
      // original inline code called those anyway, which would have thrown at
      // runtime the first time this path actually streamed a chunk. Iterate
      // the Readable directly instead; the externally-visible behavior
      // (decoded text chunks handed to onPartial, in order) is unchanged.
      const dec = new TextDecoder();
      try {
        for await (const chunk of r.body) {
          if (onPartial) onPartial(dec.decode(chunk, { stream: true }));
        }
      } catch {
        // Mid-stream read errors are swallowed here, matching the prior
        // behavior of silently ending the SSE response without surfacing
        // a fresh error status (headers/200 were already sent by the caller).
      }
      return { ok: true, status: r.status };
    }

    const rawText = await r.text();
    let raw = null;
    try { raw = JSON.parse(rawText); } catch { /* leave raw null */ }
    return {
      ok: r.ok,
      status: r.status,
      raw,
      rawText,
      text: raw?.choices?.[0]?.message?.content,
      stopReason: raw?.choices?.[0]?.finish_reason,
    };
  } catch (e) {
    if (timer) clearTimeout(timer);
    throw e;
  }
}

/**
 * Unified LLM dispatch for groq / openai / ollama / anthropic.
 *
 * @param {object} opts
 * @param {"groq"|"openai"|"ollama"|"anthropic"} opts.provider
 * @param {string} [opts.model] - defaults to LLM_MODELS[provider] (falling back to groq's model)
 * @param {Array<{role:string, content:any}>} opts.messages
 * @param {string} [opts.system] - explicit system prompt (anthropic only; merged with any
 *   role:"system" messages found in `messages`)
 * @param {string} [opts.apiKey]
 * @param {number} [opts.temperature=0.2]
 * @param {number} [opts.maxTokens=1024]
 * @param {boolean} [opts.jsonMode=false] - groq/openai/ollama: sets response_format json_object.
 *   anthropic (no tools): applies the "{" assistant-prefill trick.
 * @param {boolean} [opts.stream=false] - SSE streaming for groq/openai/ollama. Anthropic is
 *   ALWAYS non-streaming regardless of this flag (by design — do not reintroduce SSE for it).
 * @param {Array<object>} [opts.tools] - anthropic tool definitions (tool_use / function calling).
 *   When present, the JSON-prefill trick is skipped so tool_use turns aren't corrupted.
 * @param {object|string} [opts.toolChoice] - anthropic tool_choice pass-through.
 * @param {boolean} [opts.keepAlive=false] - when true and provider is "ollama", sets keep_alive:"30m".
 * @param {number} [opts.timeoutMs] - if set (and no external `signal` given), requests abort after this long.
 *   NOTE: if omitted, no timeout is applied at all (matches call sites that never had one).
 * @param {AbortSignal} [opts.signal] - external abort signal, takes precedence over timeoutMs.
 * @param {(chunk: string) => void} [opts.onPartial] - invoked with each decoded SSE chunk when streaming.
 *
 * @returns {Promise<{
 *   ok: boolean, status: number,
 *   raw?: any, rawText?: string,
 *   text?: string, toolUses?: Array<object>, stopReason?: string,
 * }>}
 *   For anthropic+tools, `raw`/`toolUses`/`stopReason` are the primary surface — callers
 *   implementing a tool-use loop should inspect `raw.content` / `stopReason` directly rather
 *   than relying on `text` (which is not coerced into a single answer in that case).
 *   For streaming calls, only `ok`/`status` are meaningful; body content is delivered via
 *   `onPartial` as it arrives, not buffered into the return value.
 */
export async function callLLM(opts) {
  const {
    provider,
    model,
    messages = [],
  } = opts;

  const resolvedModel = model || LLM_MODELS[provider] || LLM_MODELS.groq;
  const temperature = opts.temperature ?? 0.2;
  const maxTokens = opts.maxTokens ?? 1024;

  if (provider === "anthropic") {
    return callAnthropic({
      model: resolvedModel,
      messages,
      system: opts.system,
      apiKey: opts.apiKey,
      temperature,
      maxTokens,
      jsonMode: !!opts.jsonMode,
      tools: opts.tools,
      toolChoice: opts.toolChoice,
      timeoutMs: opts.timeoutMs,
      signal: opts.signal,
    });
  }

  return callOpenAiCompatible({
    provider,
    model: resolvedModel,
    messages,
    apiKey: opts.apiKey,
    temperature,
    maxTokens,
    jsonMode: !!opts.jsonMode,
    stream: !!opts.stream,
    keepAlive: !!opts.keepAlive,
    timeoutMs: opts.timeoutMs,
    signal: opts.signal,
    onPartial: opts.onPartial,
  });
}
