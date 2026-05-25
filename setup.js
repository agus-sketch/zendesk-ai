#!/usr/bin/env node
import { execSync } from "child_process";
import { existsSync, readFileSync, writeFileSync } from "fs";
import readline from "readline/promises";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ENV_PATH  = path.join(__dirname, ".env");

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

async function ask(prompt, fallback = "") {
  const hint = fallback ? ` (${fallback})` : "";
  const ans  = (await rl.question(`${prompt}${hint}: `)).trim();
  return ans || fallback;
}

async function confirm(prompt, defYes = true) {
  const tag = defYes ? "[Y/n]" : "[y/N]";
  const ans = (await rl.question(`${prompt} ${tag} `)).trim().toLowerCase();
  if (!ans) return defYes;
  return ans === "y" || ans === "yes";
}

function has(cmd) {
  try { execSync(`command -v ${cmd}`, { stdio: "ignore", shell: "/bin/sh" }); return true; }
  catch { return false; }
}

function run(cmd, opts = {}) {
  console.log(`▶ ${cmd}`);
  execSync(cmd, { stdio: "inherit", cwd: __dirname, ...opts });
}

function readEnv() {
  if (!existsSync(ENV_PATH)) return {};
  const lines = readFileSync(ENV_PATH, "utf8").split("\n");
  return Object.fromEntries(
    lines
      .filter(l => l.includes("=") && !l.startsWith("#"))
      .map(l => { const [k, ...v] = l.split("="); return [k.trim(), v.join("=").trim()]; })
  );
}

async function main() {
  console.log("\n🤖  ZAI — Zendesk AI setup\n");

  const major = Number(process.versions.node.split(".")[0]);
  if (major < 18) {
    console.error(`✗ Node ${process.versions.node} — needs Node 18+. Update at https://nodejs.org`);
    process.exit(1);
  }
  console.log(`✓ Node ${process.versions.node}`);

  if (!existsSync(path.join(__dirname, "node_modules"))) {
    console.log("\n📦 Installing dependencies…");
    run("npm install");
  } else {
    console.log("✓ Dependencies already installed");
  }

  // ── Zendesk instance ─────────────────────────────────────────────────────
  console.log("\n🔗  Zendesk instance");
  console.log("   (users enter their own email + API token in the browser UI)\n");
  const env = readEnv();

  const subdomain = await ask("  Your Zendesk subdomain (the part before .zendesk.com)", env.ZENDESK_SUBDOMAIN || "");
  if (!subdomain) {
    console.log("  ⚠ No subdomain entered — you can add ZENDESK_SUBDOMAIN= to .env later.");
  }

  writeFileSync(ENV_PATH, `ZENDESK_SUBDOMAIN=${subdomain}\nPORT=3001\n`);
  console.log(`✓ Saved to .env`);

  // ── CLI install ──────────────────────────────────────────────────────────
  if (!has("zendesk-ai")) {
    if (await confirm("\nInstall the `zendesk-ai` CLI globally so you can run it from anywhere?")) {
      try { run("npm link"); }
      catch {
        console.log("  ↳ retrying with sudo…");
        try { run("sudo npm link"); }
        catch { console.log("  ⚠ npm link failed — you can still run with `npm start`."); }
      }
    }
  } else {
    console.log("✓ `zendesk-ai` CLI already on PATH");
  }

  // ── Ollama ───────────────────────────────────────────────────────────────
  if (await confirm("\nUse Ollama (local AI) instead of Groq / OpenAI?", false)) {
    if (!has("ollama")) {
      console.log("\n⚙ Ollama is not installed.");
      const isUnix = process.platform === "darwin" || process.platform === "linux";
      if (isUnix && await confirm("  Install it now via the official script?")) {
        try { run("curl -fsSL https://ollama.com/install.sh | sh"); }
        catch { console.log("  ⚠ Install failed — try manually: https://ollama.com/download"); }
      } else if (!isUnix) {
        console.log("  Download from https://ollama.com/download, then re-run setup.");
      }
    } else {
      console.log("✓ Ollama already installed");
    }

    if (has("ollama")) {
      let installed = [];
      try {
        const out = execSync("ollama list", { encoding: "utf8" });
        installed = out.split("\n").slice(1).map(l => l.split(/\s+/)[0]).filter(Boolean);
      } catch {}

      const model = "qwen2.5:14b";
      if (!installed.includes(model)) {
        if (await confirm(`  Pull ${model} now? (~9 GB download)`)) {
          try { run(`ollama pull ${model}`); }
          catch { console.log("  ⚠ Pull failed — retry later with the same command."); }
        }
      } else {
        console.log(`✓ ${model} already installed`);
      }
    }
  }

  console.log("\n✅ Setup complete.\n");
  const start = await confirm("Start ZAI now?");
  rl.close();

  if (start) {
    if (has("zendesk-ai")) {
      run("zendesk-ai start");
    } else {
      run("npm start");
    }
    console.log("\nOpen: http://localhost:3001");
    console.log("Enter your Zendesk email + API token and your LLM key in the UI.");
  } else {
    console.log("\nStart anytime with:  zendesk-ai start   (or  npm start)");
    console.log("Then open:           http://localhost:3001");
  }
}

main().catch(e => {
  console.error(e);
  rl.close();
  process.exit(1);
});
