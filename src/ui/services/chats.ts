import { mkdir, readFile, writeFile, readdir, stat, unlink } from "fs/promises";
import { join } from "path";
import { CHATS_DIR } from "../constants";

export interface ChatMessage {
  role: string;
  text: string;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
  preview: string;
}

export interface ChatListEntry {
  id: string;
  preview: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
}

function sanitizeId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 64);
}

function chatPath(id: string): string {
  return join(CHATS_DIR, `${sanitizeId(id)}.json`);
}

function buildPreview(messages: ChatMessage[]): string {
  const first = messages.find((m) => m.role === "user");
  if (!first) return "(empty)";
  return first.text.slice(0, 80) + (first.text.length > 80 ? "..." : "");
}

export async function listChats(): Promise<ChatListEntry[]> {
  await mkdir(CHATS_DIR, { recursive: true });
  let files: string[];
  try {
    files = await readdir(CHATS_DIR);
  } catch {
    return [];
  }

  const entries = await Promise.all(
    files
      .filter((f) => f.endsWith(".json"))
      .map(async (name) => {
        const path = join(CHATS_DIR, name);
        try {
          const s = await stat(path);
          const data = JSON.parse(await readFile(path, "utf-8"));
          return {
            id: data.id || name.replace(".json", ""),
            preview: data.preview || "(empty)",
            messageCount: Array.isArray(data.messages) ? data.messages.length : 0,
            createdAt: data.createdAt || s.birthtime.toISOString(),
            updatedAt: data.updatedAt || s.mtime.toISOString(),
            mtime: s.mtimeMs,
          };
        } catch {
          return null;
        }
      })
  );

  return entries
    .filter((x): x is ChatListEntry & { mtime: number } => Boolean(x))
    .sort((a, b) => b.mtime - a.mtime)
    .map(({ mtime: _, ...rest }) => rest)
    .slice(0, 50);
}

export async function loadChat(id: string): Promise<ChatSession | null> {
  const path = chatPath(id);
  try {
    const data = JSON.parse(await readFile(path, "utf-8"));
    return {
      id: data.id || id,
      messages: Array.isArray(data.messages) ? data.messages : [],
      createdAt: data.createdAt || "",
      updatedAt: data.updatedAt || "",
      preview: data.preview || "",
    };
  } catch {
    return null;
  }
}

export async function saveChat(
  id: string,
  messages: ChatMessage[]
): Promise<ChatSession> {
  await mkdir(CHATS_DIR, { recursive: true });
  const safeId = sanitizeId(id);
  const path = chatPath(safeId);

  let createdAt: string;
  try {
    const existing = JSON.parse(await readFile(path, "utf-8"));
    createdAt = existing.createdAt || new Date().toISOString();
  } catch {
    createdAt = new Date().toISOString();
  }

  const session: ChatSession = {
    id: safeId,
    messages,
    createdAt,
    updatedAt: new Date().toISOString(),
    preview: buildPreview(messages),
  };

  await writeFile(path, JSON.stringify(session, null, 2), "utf-8");
  return session;
}

export async function deleteChat(id: string): Promise<void> {
  const path = chatPath(id);
  await unlink(path);
}
