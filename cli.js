#!/usr/bin/env node
import { spawn, execSync } from "child_process";
import { existsSync, readFileSync, writeFileSync, unlinkSync, openSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PID_FILE  = "/tmp/zendesk-ai.pid";
const LOG_FILE  = "/tmp/zendesk-ai.log";
const PORT      = process.env.PORT || 3001;
const command   = process.argv[2];

function isRunning(pid) {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

function getPid() {
  if (!existsSync(PID_FILE)) return null;
  const pid = parseInt(readFileSync(PID_FILE, "utf8").trim());
  if (isNaN(pid) || !isRunning(pid)) { try { unlinkSync(PID_FILE); } catch {} return null; }
  return pid;
}

function clearPort() {
  try { execSync(`lsof -ti :${PORT} | xargs kill -9 2>/dev/null || true`, { shell: true, stdio: "ignore" }); } catch {}
}

function start() {
  const pid = getPid();
  if (pid) { console.log(`⚠  Already running (PID ${pid})\n   → http://localhost:${PORT}`); return; }
  clearPort();
  const log    = openSync(LOG_FILE, "a");
  const server = spawn("node", [path.join(__dirname, "zendesk-agent.js")], {
    detached: true,
    stdio: ["ignore", log, log],
  });
  writeFileSync(PID_FILE, String(server.pid));
  server.unref();
  console.log(`✅ Zendesk Agent started (PID ${server.pid})\n   → http://localhost:${PORT}\n   Logs → ${LOG_FILE}`);
}

function stop() {
  const pid = getPid();
  if (!pid) { console.log("⚠  Zendesk Agent is not running"); return false; }
  try { process.kill(pid, "SIGTERM"); } catch {}
  try { unlinkSync(PID_FILE); } catch {}
  console.log(`🛑 Stopped (PID ${pid})`);
  return true;
}

function status() {
  const pid = getPid();
  if (pid) {
    console.log(`✅ Running\n   PID  → ${pid}\n   URL  → http://localhost:${PORT}\n   Logs → ${LOG_FILE}`);
  } else {
    console.log(`🔴 Not running\n   Start with: zendesk-ai start`);
  }
}

function logs() {
  if (!existsSync(LOG_FILE)) { console.log("No log file yet."); return; }
  try { execSync(`tail -50 ${LOG_FILE}`, { stdio: "inherit" }); } catch {}
}

switch (command) {
  case "start":
    start();
    break;
  case "stop":
    stop();
    break;
  case "restart":
    stop();
    await new Promise(r => setTimeout(r, 1500));
    start();
    break;
  case "status":
    status();
    break;
  case "logs":
    logs();
    break;
  case "setup":
    spawn("node", [path.join(__dirname, "setup.js")], { stdio: "inherit" }).on("exit", c => process.exit(c || 0));
    break;
  default:
    console.log("Usage: zendesk-ai start | stop | restart | status | logs | setup");
}
