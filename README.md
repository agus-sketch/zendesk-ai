# ZAI — Zendesk AI

AI-powered assistant for managing Zendesk Help Center articles. Search, read, explain, create, edit, and organize articles through a chat interface.

## Quick start

```bash
git clone <this-repo>
cd zendesk-ai
node setup.js
```

The setup wizard will:
1. Install dependencies
2. Ask for your Zendesk subdomain (e.g. `yourcompany` from `yourcompany.zendesk.com`)
3. Optionally install the `zendesk-ai` CLI globally
4. Optionally set up Ollama for local AI

Then open **http://localhost:3001** and enter:
- Your Zendesk email + API token (Admin → Apps & Integrations → Zendesk API)
- Your LLM key (Groq, OpenAI, or Anthropic) — or use Ollama locally

## CLI

```bash
zendesk-ai start     # start in background
zendesk-ai stop      # stop
zendesk-ai restart   # restart
zendesk-ai status    # check if running
zendesk-ai logs      # tail logs
zendesk-ai setup     # re-run setup (change subdomain, reinstall CLI, etc.)
```

## What you can ask

- *"Find articles about dream rate"*
- *"Explain the lead workflow article"*
- *"Show all articles in the Tracked Events section"*
- *"Create an article about how to reset a password"*
- *"Update the rate table article, add a note about refinancing"*
- *"Publish the draft article about contact forms"*
- *"What steps does the SMS text article mention?"*

## LLM providers

| Provider | Notes |
|----------|-------|
| Groq | Fast, free tier available |
| OpenAI | Reliable |
| Anthropic (Claude) | Best instruction-following |
| Ollama | Local & private, no API key needed |

## Vercel deployment

Set `ZENDESK_SUBDOMAIN` as an environment variable in your Vercel project. Users authenticate with their own email + API token in the UI — nothing sensitive lives on the server.
