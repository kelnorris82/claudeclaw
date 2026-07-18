import { mkdir, readFile, writeFile } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";
import { getSession, createSession, incrementTurn, markCompactWarned } from "./sessions";
import {
  getThreadSession,
  createThreadSession,
  incrementThreadTurn,
  markThreadCompactWarned,
} from "./sessionManager";
import { getSettings, type ModelConfig, type SecurityConfig } from "./config";
import { buildClockPromptPrefix } from "./timezone";
import { selectModel } from "./model-router";
import { loadAgent } from "./agents";
import { resolveWorkspaceSecret } from "./workspaceConfig";
import {
  appendEntry,
  formatEntriesForPrompt,
  JournalFilter,
  loadRecentEntriesForChat,
  setCursor,
  rotateIfLarge,
} from "./journal";
import { resolveStateDir } from "./paths";

const LOGS_DIR = join(resolveStateDir(), "logs");
// Resolve prompts relative to the Caravel installation, not the project dir
const PROMPTS_DIR = join(import.meta.dir, "..", "prompts");
const HEARTBEAT_PROMPT_FILE = join(PROMPTS_DIR, "heartbeat", "HEARTBEAT.md");
// Project-level prompt overrides live here (gitignored, user-owned)
const PROJECT_PROMPTS_DIR = join(resolveStateDir(), "prompts");
const PROJECT_CLAUDE_MD = join(process.cwd(), "CLAUDE.md");
const LEGACY_PROJECT_CLAUDE_MD = join(process.cwd(), ".claude", "CLAUDE.md");
const CARAVEL_BLOCK_START = "<!-- caravel:managed:start -->";
const CARAVEL_BLOCK_END = "<!-- caravel:managed:end -->";
// Pre-rebrand marker pair. Still recognised when merging an existing CLAUDE.md
// so installs created before the caravel rename get their block replaced in
// place rather than having a second one appended.
const LEGACY_BLOCK_START = "<!-- claudeclaw:managed:start -->";
const LEGACY_BLOCK_END = "<!-- claudeclaw:managed:end -->";

/**
 * Compact configuration.
 * COMPACT_WARN_THRESHOLD: notify user that context is getting large.
 * COMPACT_TIMEOUT_ENABLED: whether to auto-compact on timeout (exit 124).
 */
const COMPACT_WARN_THRESHOLD = 25;
const COMPACT_TIMEOUT_ENABLED = true;

export type CompactEvent =
  | { type: "warn"; turnCount: number }
  | { type: "auto-compact-start" }
  | { type: "auto-compact-done"; success: boolean }
  | { type: "auto-compact-retry"; success: boolean; stdout: string; stderr: string; exitCode: number };

type CompactEventListener = (event: CompactEvent) => void;
const compactListeners: CompactEventListener[] = [];

/** Register a listener for compact-related events (warnings, auto-compact notifications). */
export function onCompactEvent(listener: CompactEventListener): void {
  compactListeners.push(listener);
}

function emitCompactEvent(event: CompactEvent): void {
  for (const listener of compactListeners) {
    try { listener(event); } catch {}
  }
}

export interface RunResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

const RATE_LIMIT_PATTERN = /you.ve hit your limit|out of extra usage/i;

// Serial queue — prevents concurrent --resume on the same session
// Global queue for non-thread messages (backward compatible)
let globalQueue: Promise<unknown> = Promise.resolve();
// Per-thread queues — each thread runs independently in parallel
const threadQueues = new Map<string, Promise<unknown>>();

function enqueue<T>(fn: () => Promise<T>, threadId?: string): Promise<T> {
  if (threadId) {
    const current = threadQueues.get(threadId) ?? Promise.resolve();
    const task = current.then(fn, fn);
    threadQueues.set(threadId, task.catch(() => {}));
    return task;
  }
  const task = globalQueue.then(fn, fn);
  globalQueue = task.catch(() => {});
  return task;
}

function extractRateLimitMessage(stdout: string, stderr: string): string | null {
  const candidates = [stdout, stderr];
  for (const text of candidates) {
    const trimmed = text.trim();
    if (trimmed && RATE_LIMIT_PATTERN.test(trimmed)) return trimmed;
  }
  return null;
}

function sameModelConfig(a: ModelConfig, b: ModelConfig): boolean {
  return a.model.trim().toLowerCase() === b.model.trim().toLowerCase() && a.api.trim() === b.api.trim();
}

function hasModelConfig(value: ModelConfig): boolean {
  return value.model.trim().length > 0 || value.api.trim().length > 0;
}

function isNotFoundError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const code = (error as { code?: unknown }).code;
  if (code === "ENOENT") return true;
  const message = String((error as { message?: unknown }).message ?? "");
  return /enoent|no such file or directory/i.test(message);
}

function buildChildEnv(
  baseEnv: Record<string, string>,
  model: string,
  api: string,
  threadId?: string,
): Record<string, string> {
  const childEnv: Record<string, string> = { ...baseEnv };
  const normalizedModel = model.trim().toLowerCase();

  if (api.trim()) childEnv.ANTHROPIC_AUTH_TOKEN = api.trim();

  if (normalizedModel === "glm") {
    childEnv.ANTHROPIC_BASE_URL = "https://api.z.ai/api/anthropic";
    childEnv.API_TIMEOUT_MS = "3000000";
  }

  if (threadId) {
    childEnv.CARAVEL_CHAT_ID = threadId;
    childEnv.CLAUDECLAW_CHAT_ID = threadId; // backward compat for old task.mjs templates
  }

  return childEnv;
}

/** Default timeout for a single Claude Code invocation (5 minutes). */
const CLAUDE_TIMEOUT_MS = 5 * 60 * 1000;

async function runClaudeOnce(
  baseArgs: string[],
  model: string,
  api: string,
  baseEnv: Record<string, string>,
  timeoutMs: number = CLAUDE_TIMEOUT_MS
): Promise<{ rawStdout: string; stderr: string; exitCode: number }> {
  const args = [...baseArgs];
  const normalizedModel = model.trim().toLowerCase();
  if (model.trim() && normalizedModel !== "glm") args.push("--model", model.trim());

  const proc = Bun.spawn(args, {
    stdout: "pipe",
    stderr: "pipe",
    env: buildChildEnv(baseEnv, model, api),
  });

  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(`Claude session timed out after ${timeoutMs / 1000}s`)), timeoutMs);
  });

  try {
    const [rawStdout, stderr] = await Promise.race([
      Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]),
      timeoutPromise,
    ]) as [string, string];
    await proc.exited;

    return {
      rawStdout,
      stderr,
      exitCode: proc.exitCode ?? 1,
    };
  } catch (err) {
    // Kill the hung process
    try { proc.kill("SIGTERM"); } catch {}
    setTimeout(() => { try { proc.kill("SIGKILL"); } catch {} }, 5000);

    const message = err instanceof Error ? err.message : String(err);
    console.error(`[${new Date().toLocaleTimeString()}] ${message}`);

    return {
      rawStdout: "",
      stderr: message,
      exitCode: 124,
    };
  }
}

const PROJECT_DIR = process.cwd();

const DIR_SCOPE_PROMPT = [
  `CRITICAL SECURITY CONSTRAINT: You are scoped to the project directory: ${PROJECT_DIR}`,
  "You MUST NOT read, write, edit, or delete any file outside this directory.",
  "You MUST NOT run bash commands that modify anything outside this directory (no cd /, no /etc, no ~/, no ../.. escapes).",
  "If a request requires accessing files outside the project, refuse and explain why.",
].join("\n");

export async function ensureProjectClaudeMd(): Promise<void> {
  // Preflight-only initialization: never rewrite an existing project CLAUDE.md.
  if (existsSync(PROJECT_CLAUDE_MD)) return;
  // When the project uses agent profiles (./agents/), identity lives per-agent
  // and there is intentionally no root CLAUDE.md. Don't regenerate one.
  if (existsSync(join(process.cwd(), "agents"))) return;

  const promptContent = (await loadPrompts()).trim();
  const managedBlock = [
    CARAVEL_BLOCK_START,
    promptContent,
    CARAVEL_BLOCK_END,
  ].join("\n");

  let content = "";

  if (existsSync(LEGACY_PROJECT_CLAUDE_MD)) {
    try {
      const legacy = await readFile(LEGACY_PROJECT_CLAUDE_MD, "utf8");
      content = legacy.trim();
    } catch (e) {
      console.error(`[${new Date().toLocaleTimeString()}] Failed to read legacy .claude/CLAUDE.md:`, e);
      return;
    }
  }

  const normalized = content.trim();

  // Match the current marker pair first, then the pre-rebrand one. Checking the
  // legacy pair matters: without it an older CLAUDE.md would gain a second
  // managed block instead of having its existing one replaced.
  const markerPairs = [
    [CARAVEL_BLOCK_START, CARAVEL_BLOCK_END],
    [LEGACY_BLOCK_START, LEGACY_BLOCK_END],
  ];
  const existingPair = markerPairs.find(
    ([start, end]) => normalized.includes(start) && normalized.includes(end)
  );

  const merged = existingPair
    ? `${normalized.replace(
        new RegExp(`${existingPair[0]}[\\s\\S]*?${existingPair[1]}`, "m"),
        managedBlock
      )}\n`
    : normalized
      ? `${normalized}\n\n${managedBlock}\n`
      : `${managedBlock}\n`;

  try {
    await writeFile(PROJECT_CLAUDE_MD, merged, "utf8");
  } catch (e) {
    console.error(`[${new Date().toLocaleTimeString()}] Failed to write project CLAUDE.md:`, e);
  }
}

function buildSecurityArgs(security: SecurityConfig): string[] {
  const args: string[] = ["--dangerously-skip-permissions"];

  switch (security.level) {
    case "locked":
      args.push("--tools", "Read,Grep,Glob");
      break;
    case "strict":
      args.push("--disallowedTools", "Bash,WebSearch,WebFetch");
      break;
    case "moderate":
      // all tools available, scoped to project dir via system prompt
      break;
    case "unrestricted":
      // all tools, no directory restriction
      break;
  }

  if (security.allowedTools.length > 0) {
    args.push("--allowedTools", security.allowedTools.join(" "));
  }
  if (security.disallowedTools.length > 0) {
    args.push("--disallowedTools", security.disallowedTools.join(" "));
  }

  return args;
}

/** Load and concatenate all prompt files from the prompts/ directory. */
async function loadPrompts(): Promise<string> {
  const selectedPromptFiles = [
    join(PROMPTS_DIR, "IDENTITY.md"),
    join(PROMPTS_DIR, "USER.md"),
    join(PROMPTS_DIR, "SOUL.md"),
  ];
  const parts: string[] = [];

  for (const file of selectedPromptFiles) {
    try {
      const content = await Bun.file(file).text();
      if (content.trim()) parts.push(content.trim());
    } catch (e) {
      console.error(`[${new Date().toLocaleTimeString()}] Failed to read prompt file ${file}:`, e);
    }
  }

  return parts.join("\n\n");
}

/**
 * Load the heartbeat prompt template.
 * Project-level override takes precedence: place a file at
 * .claude/caravel/prompts/HEARTBEAT.md to fully replace the built-in template.
 */
export async function loadHeartbeatPromptTemplate(): Promise<string> {
  const projectOverride = join(PROJECT_PROMPTS_DIR, "HEARTBEAT.md");
  for (const file of [projectOverride, HEARTBEAT_PROMPT_FILE]) {
    try {
      const content = await Bun.file(file).text();
      if (content.trim()) return content.trim();
    } catch (e) {
      if (!isNotFoundError(e)) {
        console.warn(`[${new Date().toLocaleTimeString()}] Failed to read heartbeat prompt file ${file}:`, e);
      }
    }
  }
  return "";
}

/** Run /compact on the current session to reduce context size. */
export async function runCompact(
  sessionId: string,
  model: string,
  api: string,
  baseEnv: Record<string, string>,
  securityArgs: string[],
  timeoutMs: number
): Promise<boolean> {
  const compactArgs = [
    "claude", "-p", "/compact",
    "--output-format", "text",
    "--resume", sessionId,
    ...securityArgs,
  ];
  console.log(`[${new Date().toLocaleTimeString()}] Running /compact on session ${sessionId.slice(0, 8)}...`);
  const result = await runClaudeOnce(compactArgs, model, api, baseEnv, timeoutMs);
  const success = result.exitCode === 0;
  console.log(`[${new Date().toLocaleTimeString()}] Compact ${success ? "succeeded" : `failed (exit ${result.exitCode})`}`);
  return success;
}

/**
 * High-level compact: resolves session + settings internally.
 * Returns { success, message }.
 */
export async function compactCurrentSession(): Promise<{ success: boolean; message: string }> {
  const existing = await getSession();
  if (!existing) return { success: false, message: "No active session to compact." };

  const settings = getSettings();
  const securityArgs = buildSecurityArgs(settings.security);
  const { CLAUDECODE: _, ...cleanEnv } = process.env;
  const baseEnv = { ...cleanEnv } as Record<string, string>;
  const timeoutMs = (settings as any).sessionTimeoutMs || CLAUDE_TIMEOUT_MS;

  const ok = await runCompact(
    existing.sessionId,
    settings.model,
    settings.api,
    baseEnv,
    securityArgs,
    timeoutMs
  );

  return ok
    ? { success: true, message: `✅ Session compact complete (${existing.sessionId.slice(0, 8)})` }
    : { success: false, message: `❌ Compact failed (${existing.sessionId.slice(0, 8)})` };
}

async function execClaude(name: string, prompt: string, threadId?: string): Promise<RunResult> {
  await mkdir(LOGS_DIR, { recursive: true });

  const existing = threadId
    ? await getThreadSession(threadId)
    : await getSession();
  const isNew = !existing;
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const logFile = join(LOGS_DIR, `${name}-${timestamp}.log`);

  const settings = getSettings();
  const { security, model, api, fallback, agentic } = settings;

  // Determine which model to use based on agentic routing
  let primaryConfig: ModelConfig;
  let taskType = "unknown";
  let routingReasoning = "";

  if (agentic.enabled) {
    const routing = selectModel(prompt, agentic.modes, agentic.defaultMode);
    primaryConfig = { model: routing.model, api };
    taskType = routing.taskType;
    routingReasoning = routing.reasoning;
    console.log(
      `[${new Date().toLocaleTimeString()}] Agentic routing: ${routing.taskType} → ${routing.model} (${routing.reasoning})`
    );
  } else {
    primaryConfig = { model, api };
  }

  const fallbackConfig: ModelConfig = {
    model: fallback?.model ?? "",
    api: fallback?.api ?? "",
  };
  const securityArgs = buildSecurityArgs(security);
  const timeoutMs = (settings as any).sessionTimeoutMs || CLAUDE_TIMEOUT_MS;

  console.log(
    `[${new Date().toLocaleTimeString()}] Running: ${name} (${isNew ? "new session" : `resume ${existing.sessionId.slice(0, 8)}`}, security: ${security.level})`
  );

  // New session: use json output to capture Claude's session_id
  // Resumed session: use text output with --resume
  const outputFormat = isNew ? "json" : "text";
  const args = ["claude", "-p", prompt, "--output-format", outputFormat, ...securityArgs];

  if (!isNew) {
    args.push("--resume", existing.sessionId);
  }

  // Build the appended system prompt: prompt files + directory scoping
  // This is passed on EVERY invocation (not just new sessions) because
  // --append-system-prompt does not persist across --resume.
  const promptContent = await loadPrompts();
  const appendParts: string[] = [
    "You are running inside Caravel.",
  ];
  if (promptContent) appendParts.push(promptContent);

  // Load the project's CLAUDE.md if it exists
  if (existsSync(PROJECT_CLAUDE_MD)) {
    try {
      const claudeMd = await Bun.file(PROJECT_CLAUDE_MD).text();
      if (claudeMd.trim()) appendParts.push(claudeMd.trim());
    } catch (e) {
      console.error(`[${new Date().toLocaleTimeString()}] Failed to read project CLAUDE.md:`, e);
    }
  }

  if (security.level !== "unrestricted") appendParts.push(DIR_SCOPE_PROMPT);
  if (appendParts.length > 0) {
    args.push("--append-system-prompt", appendParts.join("\n\n"));
  }

  // Strip CLAUDECODE env var so child claude processes don't think they're nested
  const { CLAUDECODE: _, ...cleanEnv } = process.env;
  const baseEnv = { ...cleanEnv } as Record<string, string>;

  let exec = await runClaudeOnce(args, primaryConfig.model, primaryConfig.api, baseEnv, timeoutMs);
  const primaryRateLimit = extractRateLimitMessage(exec.rawStdout, exec.stderr);
  let usedFallback = false;

  if (primaryRateLimit && hasModelConfig(fallbackConfig) && !sameModelConfig(primaryConfig, fallbackConfig)) {
    console.warn(
      `[${new Date().toLocaleTimeString()}] Claude limit reached; retrying with fallback${fallbackConfig.model ? ` (${fallbackConfig.model})` : ""}...`
    );
    exec = await runClaudeOnce(args, fallbackConfig.model, fallbackConfig.api, baseEnv, timeoutMs);
    usedFallback = true;
  }

  const rawStdout = exec.rawStdout;
  const stderr = exec.stderr;
  const exitCode = exec.exitCode;
  let stdout = rawStdout;
  let sessionId = existing?.sessionId ?? "unknown";
  const rateLimitMessage = extractRateLimitMessage(rawStdout, stderr);

  if (rateLimitMessage) {
    stdout = rateLimitMessage;
  }

  // For new sessions, parse the JSON to extract session_id and result text
  if (!rateLimitMessage && isNew && exitCode === 0) {
    try {
      const json = JSON.parse(rawStdout);
      sessionId = json.session_id;
      stdout = json.result ?? "";
      // Save the real session ID from Claude Code
      if (threadId) {
        await createThreadSession(threadId, sessionId);
        console.log(`[${new Date().toLocaleTimeString()}] Thread session created: ${sessionId} (thread ${threadId.slice(0, 8)})`);
      } else {
        await createSession(sessionId);
        console.log(`[${new Date().toLocaleTimeString()}] Session created: ${sessionId}`);
      }
    } catch (e) {
      console.error(`[${new Date().toLocaleTimeString()}] Failed to parse session from Claude output:`, e);
    }
  }

  const result: RunResult = {
    stdout,
    stderr,
    exitCode,
  };

  const output = [
    `# ${name}`,
    `Date: ${new Date().toISOString()}`,
    `Session: ${sessionId} (${isNew ? "new" : "resumed"})`,
    `Model config: ${usedFallback ? "fallback" : "primary"}`,
    ...(agentic.enabled ? [`Task type: ${taskType}`, `Routing: ${routingReasoning}`] : []),
    `Prompt: ${prompt}`,
    `Exit code: ${result.exitCode}`,
    "",
    "## Output",
    stdout,
    ...(stderr ? ["## Stderr", stderr] : []),
  ].join("\n");

  await Bun.write(logFile, output);
  console.log(`[${new Date().toLocaleTimeString()}] Done: ${name} → ${logFile}`);

  // --- Auto-compact on timeout (exit 124) ---
  if (COMPACT_TIMEOUT_ENABLED && exitCode === 124 && !isNew && existing) {
    emitCompactEvent({ type: "auto-compact-start" });
    const compactOk = await runCompact(
      existing.sessionId,
      primaryConfig.model,
      primaryConfig.api,
      baseEnv,
      securityArgs,
      timeoutMs
    );
    emitCompactEvent({ type: "auto-compact-done", success: compactOk });

    if (compactOk) {
      console.log(`[${new Date().toLocaleTimeString()}] Retrying ${name} after compact...`);
      const retryExec = await runClaudeOnce(args, primaryConfig.model, primaryConfig.api, baseEnv, timeoutMs);
      const retryResult: RunResult = {
        stdout: retryExec.rawStdout,
        stderr: retryExec.stderr,
        exitCode: retryExec.exitCode,
      };
      emitCompactEvent({
        type: "auto-compact-retry",
        success: retryExec.exitCode === 0,
        stdout: retryResult.stdout,
        stderr: retryResult.stderr,
        exitCode: retryResult.exitCode,
      });

      if (retryExec.exitCode === 0) {
        const count = threadId ? await incrementThreadTurn(threadId) : await incrementTurn();
        console.log(`[${new Date().toLocaleTimeString()}] Turn count: ${count} (after compact + retry)`);
      }
      return retryResult;
    }
  }

  // --- Turn tracking & compact warning ---
  if (exitCode === 0 && !isNew) {
    const turnCount = threadId ? await incrementThreadTurn(threadId) : await incrementTurn();
    console.log(`[${new Date().toLocaleTimeString()}] Turn count: ${turnCount}${threadId ? ` (thread ${threadId.slice(0, 8)})` : ""}`);

    if (turnCount >= COMPACT_WARN_THRESHOLD && existing && !existing.compactWarned) {
      if (threadId) {
        await markThreadCompactWarned(threadId);
      } else {
        await markCompactWarned();
      }
      emitCompactEvent({ type: "warn", turnCount });
    }
  }

  return result;
}

export async function run(name: string, prompt: string, threadId?: string): Promise<RunResult> {
  return enqueue(() => execClaude(name, prompt, threadId), threadId);
}

async function streamClaude(
  name: string,
  prompt: string,
  onChunk: (text: string) => void,
  onUnblock: () => void,
  abortSignal?: AbortSignal,
  threadId?: string,
  agentId?: string,
  // Optional out-of-band diagnostic channel. Receives the child's stderr,
  // the final `result` event text (always — even when assistant text was
  // emitted), and a non-zero exit marker. NOT shown to chat users; the
  // multi-agent runner supplies it to spot usage-limit signatures that
  // never reach the assistant-text stream (stderr-only messages, error
  // result events after prior output, immediate non-zero exits). Chat
  // callers omit it and behaviour is unchanged.
  onDiagnostic?: (text: string) => void
): Promise<void> {
  await mkdir(LOGS_DIR, { recursive: true });

  // Mirror execClaude's threadId handling: per-thread sessions when a threadId
  // is supplied (web chat, Discord), global session otherwise (heartbeat, CLI).
  const existing = threadId
    ? await getThreadSession(threadId)
    : await getSession();
  const { security, model, api } = getSettings();
  const securityArgs = buildSecurityArgs(security);

  // Load agent profile if one is selected for this chat.
  const agent = agentId ? await loadAgent(agentId) : null;

  // stream-json gives us events as they happen — text before tool calls,
  // so we can unblock the UI as soon as Claude acknowledges, not after sub-agents finish.
  // --verbose is required for stream-json to produce output in -p (print) mode.
  const args = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", ...securityArgs];

  if (existing) args.push("--resume", existing.sessionId);

  const promptContent = await loadPrompts();
  const appendParts: string[] = ["You are running inside Caravel."];
  if (promptContent) appendParts.push(promptContent);

  // CLAUDE.md selection:
  //   - Agent selected: use the agent's CLAUDE.md + agent rules.
  //   - No agent: fall back to the project CLAUDE.md (upstream single-agent
  //     projects that don't have an agents/ directory).
  if (agent) {
    if (agent.claudeMd) appendParts.push(agent.claudeMd);
    if (agent.rules) appendParts.push(agent.rules);
  } else if (existsSync(PROJECT_CLAUDE_MD)) {
    try {
      const claudeMd = await Bun.file(PROJECT_CLAUDE_MD).text();
      if (claudeMd.trim()) appendParts.push(claudeMd.trim());
    } catch {}
  }

  // Inject recent shared learnings from other chats with the same agent.
  // No-op when no agent or no chat thread (heartbeat / one-shot CLI runs).
  let injectedJournalEntries = 0;
  let latestInjectedTs: string | null = null;
  if (agent && threadId) {
    const recent = await loadRecentEntriesForChat(agent.manifest.name, threadId);
    if (recent.length > 0) {
      appendParts.push(formatEntriesForPrompt(recent));
      injectedJournalEntries = recent.length;
      latestInjectedTs = recent[recent.length - 1]?.ts ?? null;
    }
  }

  if (security.level !== "unrestricted") appendParts.push(DIR_SCOPE_PROMPT);
  if (appendParts.length > 0) {
    args.push("--append-system-prompt", appendParts.join("\n\n"));
  }

  // Agent manifest may override the model.
  const effectiveModel = agent?.manifest.model ?? model;
  const normalizedModel = effectiveModel.trim().toLowerCase();
  if (effectiveModel.trim() && normalizedModel !== "glm") args.push("--model", effectiveModel.trim());

  const { CLAUDECODE: _, ...cleanEnv } = process.env;
  const childEnv = buildChildEnv(cleanEnv as Record<string, string>, effectiveModel, api, threadId);

  // Per-agent provider routing: if the manifest sets apiBaseUrl, point the
  // child Claude CLI at an Anthropic-compatible proxy (LiteLLM, etc.) instead
  // of api.anthropic.com. The proxy's auth token is sourced — in order — from
  // apiKeyConfig (a dotted path into .claude/config.json) then apiKeyEnv (a
  // daemon env var). Config is preferred so secrets live in the workspace
  // config, not the daemon environment.
  if (agent?.manifest.apiBaseUrl) {
    childEnv.ANTHROPIC_BASE_URL = agent.manifest.apiBaseUrl;
    let proxyToken: string | null = null;
    if (agent.manifest.apiKeyConfig) {
      proxyToken = await resolveWorkspaceSecret(agent.manifest.apiKeyConfig);
    }
    if (!proxyToken && agent.manifest.apiKeyEnv) {
      proxyToken = process.env[agent.manifest.apiKeyEnv]?.trim() || null;
    }
    if (proxyToken) childEnv.ANTHROPIC_AUTH_TOKEN = proxyToken;
    // Safeguard: when routing to a non-Anthropic provider (z.ai, OpenRouter,
    // LiteLLM, …) a stray ANTHROPIC_API_KEY in the daemon env can make the
    // child Claude CLI silently fall back to api.anthropic.com and throw
    // confusing model-not-found errors. Blank it so only ANTHROPIC_AUTH_TOKEN
    // + ANTHROPIC_BASE_URL above are honoured.
    childEnv.ANTHROPIC_API_KEY = "";
  }

  const threadTag = threadId ? ` thread: ${threadId.slice(0, 8)},` : "";
  const agentTag = agent ? ` agent: ${agent.manifest.name},` : "";
  const journalTag = injectedJournalEntries > 0 ? ` shared-notes: ${injectedJournalEntries},` : "";
  console.log(`[${new Date().toLocaleTimeString()}] Running: ${name} (stream-json,${threadTag}${agentTag}${journalTag} session: ${existing?.sessionId?.slice(0, 8) ?? "new"})`);

  const proc = Bun.spawn(args, {
    stdout: "pipe",
    stderr: "pipe",
    env: childEnv,
  });

  // Drain stderr concurrently into the diagnostic channel. The Claude CLI
  // prints account-level usage-limit messages ("You've hit your limit ·
  // resets …") and other fatal errors to stderr and exits — none of which
  // reach the stdout stream-json reader below. Without this, an immediate
  // limit-hit looks like a clean empty turn to the runner. Reading stderr
  // also prevents a full-pipe deadlock on a chatty child.
  const drainStderr = (async () => {
    if (!onDiagnostic || !proc.stderr) return;
    try {
      const sr = (proc.stderr as ReadableStream<Uint8Array>).getReader();
      const sd = new TextDecoder();
      while (true) {
        const { done, value } = await sr.read();
        if (done) break;
        const chunk = sd.decode(value, { stream: true });
        if (chunk) onDiagnostic(chunk);
      }
    } catch {
      // stderr drain is best-effort; never let it break the turn.
    }
  })();

  let aborted = false;
  const onAbort = () => {
    aborted = true;
    try { proc.kill(); } catch {}
  };
  if (abortSignal) {
    if (abortSignal.aborted) onAbort();
    else abortSignal.addEventListener("abort", onAbort, { once: true });
  }

  const reader = proc.stdout.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let unblocked = false;
  let textEmitted = false;

  // Journal filter — strips <journal …>…</journal> directives from any text
  // emitted to the UI and collects the matched entries for persistence after
  // the response completes. Only active for agent-tagged chats.
  const journalActive = !!(agent && threadId);
  const filter = journalActive ? new JournalFilter() : null;
  const emit = (text: string) => {
    if (!text) return;
    const cleaned = filter ? filter.feed(text) : text;
    if (cleaned) onChunk(cleaned);
  };

  const maybeUnblock = () => {
    if (!unblocked) {
      unblocked = true;
      onUnblock();
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // Parse complete newline-delimited JSON events
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const event = JSON.parse(trimmed) as Record<string, unknown>;

        if (event.type === "system" && (event.subtype === "init" || event.session_id)) {
          // Capture session ID for new sessions
          const sid = event.session_id as string | undefined;
          if (sid && !existing) {
            if (threadId) {
              await createThreadSession(threadId, sid);
              console.log(`[${new Date().toLocaleTimeString()}] Thread session created (stream-json): ${threadId.slice(0, 8)} → ${sid.slice(0, 8)}`);
            } else {
              await createSession(sid);
              console.log(`[${new Date().toLocaleTimeString()}] Session created (stream-json): ${sid}`);
            }
          }
        } else if (event.type === "assistant") {
          // Text and tool_use blocks from the assistant
          type ContentBlock = { type: string; text?: string; id?: string; name?: string; input?: Record<string, unknown> };
          const msg = event.message as { content?: ContentBlock[] } | undefined;
          const blocks = msg?.content ?? [];
          let hasActivity = false;
          for (const block of blocks) {
            if (block.type === "text" && block.text) {
              emit(block.text);
              textEmitted = true;
              hasActivity = true;
            } else if (block.type === "tool_use") {
              hasActivity = true;
            }
          }
          if (hasActivity) maybeUnblock();
        } else if (event.type === "tool_use") {
          // Top-level tool_use event (some stream-json versions) — unblock the UI
          maybeUnblock();
        } else if (event.type === "result") {
          // Final result event — emit text as fallback if no assistant text was seen
          const ev = event as Record<string, unknown>;
          const resultText = ev.result as string | undefined;
          if (resultText && !textEmitted) {
            emit(resultText);
          }
          // Always surface the result text + error markers to the diagnostic
          // channel, even when assistant text was already emitted. A
          // usage-limit hit after prior output lands here (subtype/is_error
          // + the limit message in `result`) and would otherwise be dropped
          // — exactly the gap that mis-marked long-running workers
          // failed:other instead of waiting:on:limits.
          if (onDiagnostic) {
            const isError = ev.is_error === true;
            const subtype = typeof ev.subtype === "string" ? ev.subtype : "";
            if (resultText) onDiagnostic(`[result] ${resultText}`);
            if (isError || (subtype && subtype !== "success")) {
              onDiagnostic(`[result is_error=${isError} subtype=${subtype}]`);
            }
          }
          maybeUnblock();
        }
      } catch {}
    }
  }

  const exitCode = await proc.exited;
  await drainStderr; // ensure all stderr is delivered before we evaluate
  // A non-zero exit with no useful stdout is the other usage-limit shape
  // (the CLI dies immediately on a hard cap). Mark it on the diagnostic
  // channel so the runner can distinguish "limit/crash" from "clean turn".
  if (onDiagnostic && exitCode !== 0 && !aborted) {
    onDiagnostic(`[claude exited ${exitCode}]`);
  }
  if (abortSignal) abortSignal.removeEventListener("abort", onAbort);
  // Flush any held-back text from the journal filter (e.g. trailing chars
  // we conservatively buffered while watching for a `<journal` tag that
  // never materialised).
  if (filter) {
    const tail = filter.flush();
    if (tail) onChunk(tail);
  }
  // Ensure unblock fires even if something unexpected happened
  maybeUnblock();

  if (aborted) {
    console.log(`[${new Date().toLocaleTimeString()}] Interrupted: ${name}`);
    return;
  }

  console.log(`[${new Date().toLocaleTimeString()}] Done: ${name}`);

  // Persist any journal entries the agent emitted in this turn, then advance
  // this chat's read cursor past whatever we injected so we don't show the
  // same shared notes again.
  if (filter && agent && threadId) {
    const drafts = filter.drainEntries();
    if (drafts.length > 0) {
      const ts = new Date().toISOString();
      for (const d of drafts) {
        await appendEntry(agent.manifest.name, {
          ts,
          chatId: threadId,
          kind: d.kind,
          text: d.text,
          ...(d.tags ? { tags: d.tags } : {}),
        });
      }
      console.log(`[${new Date().toLocaleTimeString()}] Journal: ${drafts.length} entr${drafts.length === 1 ? "y" : "ies"} appended (${agent.manifest.name})`);
      try { await rotateIfLarge(agent.manifest.name); } catch {}
    }
    if (latestInjectedTs) {
      try { await setCursor(threadId, latestInjectedTs); } catch {}
    }
  }

  // Mirror execClaude's turn tracking + compact warning. Only count turns that
  // resumed an existing session — a brand-new session's first turn is implicit.
  if (exitCode === 0 && existing) {
    const turnCount = threadId ? await incrementThreadTurn(threadId) : await incrementTurn();
    console.log(`[${new Date().toLocaleTimeString()}] Turn count: ${turnCount}${threadId ? ` (thread ${threadId.slice(0, 8)})` : ""}`);

    if (turnCount >= COMPACT_WARN_THRESHOLD && !existing.compactWarned) {
      if (threadId) {
        await markThreadCompactWarned(threadId);
      } else {
        await markCompactWarned();
      }
      emitCompactEvent({ type: "warn", turnCount });
    }
  }
}

export async function streamUserMessage(
  name: string,
  prompt: string,
  onChunk: (text: string) => void,
  onUnblock: () => void,
  abortSignal?: AbortSignal,
  threadId?: string,
  agentId?: string,
  onDiagnostic?: (text: string) => void
): Promise<void> {
  // Per-thread queue when threadId is set (web chats run in parallel, matches
  // Discord). Falls back to the global queue for heartbeat/CLI calls.
  return enqueue(
    () => streamClaude(name, prefixUserMessageWithClock(prompt), onChunk, onUnblock, abortSignal, threadId, agentId, onDiagnostic),
    threadId
  );
}

function prefixUserMessageWithClock(prompt: string): string {
  try {
    const settings = getSettings();
    const prefix = buildClockPromptPrefix(new Date(), settings.timezoneOffsetMinutes);
    return `${prefix}\n${prompt}`;
  } catch {
    const prefix = buildClockPromptPrefix(new Date(), 0);
    return `${prefix}\n${prompt}`;
  }
}

export async function runUserMessage(name: string, prompt: string, threadId?: string): Promise<RunResult> {
  const ts = () => new Date().toLocaleTimeString();
  const oneLine = (s: string, n = 120) => {
    const flat = (s || "").replace(/\s+/g, " ").trim();
    return flat.length > n ? flat.slice(0, n) + "…" : flat;
  };
  const where = threadId ? `${name}/${threadId}` : name;
  console.log(`[${ts()}] [chat] ← ${where}: ${oneLine(prompt)}`);
  const result = await run(name, prefixUserMessageWithClock(prompt), threadId);
  const reply = oneLine(result?.stdout ?? "");
  console.log(`[${ts()}] [chat] → ${where}: ${reply || "(no reply)"}${result?.exitCode ? ` [exit ${result.exitCode}]` : ""}`);
  return result;
}

/**
 * Bootstrap the session: fires Claude with the system prompt so the
 * session is created immediately. No-op if a session already exists.
 */
export async function bootstrap(): Promise<void> {
  const existing = await getSession();
  if (existing) return;

  console.log(`[${new Date().toLocaleTimeString()}] Bootstrapping new session...`);
  await execClaude("bootstrap", "Wakeup, my friend!");
  console.log(`[${new Date().toLocaleTimeString()}] Bootstrap complete — session is live.`);
}
