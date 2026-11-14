const configuredApiBase = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const apiBase = configuredApiBase.replace(/\/+$/, "");

export function getApiUrl(path: string): string {
  return `${apiBase}${path}`;
}

export type DocumentItem = {
  id: string;
  filename: string;
  media_type: string;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type Citation = {
  document_id: string;
  document_name: string;
  chunk_id: string;
  page_number: number | null;
  chunk_index: number;
  excerpt: string;
};

export type ChunkPreview = {
  document_id: string;
  document_name: string;
  chunk_id: string;
  chunk_index: number;
  page_number: number | null;
  content: string;
};

export type ChatImage = {
  document_id: string;
  document_name: string;
  image_id: string;
  page_number: number;
  filename: string;
  media_type: string;
  image_url: string;
  cited: boolean;
  context_excerpt: string | null;
};

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  images: ChatImage[];
  conversation_id: string;
  retrieval_metadata: Record<string, unknown>;
  latency_ms: number;
};

export type ConversationSummary = {
  id: string;
  title: string;
  message_count: number;
  preview: string;
  created_at: string;
  updated_at: string;
};

export type SavedMessageItem = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
};

export type ConversationDetail = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: SavedMessageItem[];
};

export type RetrievalMode = "dense" | "bm25" | "hybrid" | "hybrid_rerank";

type ChatRequest = {
  question: string;
  conversation_id?: string;
  document_ids: string[] | null;
  retrieval_mode: RetrievalMode;
  top_k: number;
  include_images: boolean;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBase}${path}`, init);
  } catch {
    throw new Error(
      "Could not reach the API. Check that the backend is running and try again.",
    );
  }

  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    try {
      const payload: unknown = await response.json();
      if (
        typeof payload === "object" &&
        payload !== null &&
        "detail" in payload
      ) {
        const detail = payload.detail;
        if (typeof detail === "string") {
          message = detail;
        } else if (Array.isArray(detail)) {
          message = detail
            .map((item) =>
              typeof item === "object" && item !== null && "msg" in item
                ? String(item.msg)
                : String(item),
            )
            .join(" ");
        }
      }
    } catch {
      // Keep the HTTP status message when the server does not return JSON.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<{ status: string }> {
  return request("/api/health");
}

export function listDocuments(): Promise<DocumentItem[]> {
  return request("/api/documents");
}

export function getChunkPreview(
  documentId: string,
  chunkId: string,
): Promise<ChunkPreview> {
  return request(
    `/api/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}`,
  );
}

export function uploadDocument(file: File): Promise<DocumentItem> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/documents", { method: "POST", body: form });
}

export function deleteDocument(documentId: string): Promise<void> {
  return request(`/api/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
}

export function listConversations(): Promise<ConversationSummary[]> {
  return request("/api/chat/conversations");
}

export function getConversation(
  conversationId: string,
): Promise<ConversationDetail> {
  return request(
    `/api/chat/conversations/${encodeURIComponent(conversationId)}`,
  );
}

export function deleteConversation(conversationId: string): Promise<void> {
  return request(
    `/api/chat/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" },
  );
}

export function sendQuestion(payload: ChatRequest): Promise<ChatResponse> {
  return request("/api/chat/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function streamQuestion(
  payload: ChatRequest,
  onAnswerDelta: (delta: string) => void,
  onAnswerReset: (answer: string) => void,
): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBase}/api/chat/query/stream`, {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Could not reach the API. Check that the backend is running and try again.",
    );
  }

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  if (!response.body) {
    throw new Error("The server did not provide a streaming response.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completion: ChatResponse | null = null;

  function processEvent(block: string): void {
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length === 0) return;
    const data: unknown = JSON.parse(dataLines.join("\n"));
    if (eventName === "delta") {
      if (
        typeof data === "object" &&
        data !== null &&
        "text" in data &&
        typeof data.text === "string"
      ) {
        onAnswerDelta(data.text);
      }
    } else if (eventName === "reset") {
      if (
        typeof data === "object" &&
        data !== null &&
        "text" in data &&
        typeof data.text === "string"
      ) {
        onAnswerReset(data.text);
      }
    } else if (eventName === "done") {
      completion = data as ChatResponse;
    } else if (eventName === "error") {
      const detail =
        typeof data === "object" &&
        data !== null &&
        "detail" in data &&
        typeof data.detail === "string"
          ? data.detail
          : "The question could not be answered.";
      throw new Error(detail);
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        processEvent(buffer.slice(0, boundary).replace(/\r$/, ""));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
      if (done) break;
    }
    if (buffer.trim()) processEvent(buffer);
  } finally {
    reader.releaseLock();
  }
  if (!completion) {
    throw new Error("The streaming response ended before the answer was complete.");
  }
  return completion;
}

async function readErrorMessage(response: Response): Promise<string> {
  let message = `Request failed (HTTP ${response.status}).`;
  try {
    const payload: unknown = await response.json();
    if (
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof payload.detail === "string"
    ) {
      message = payload.detail;
    }
  } catch {
    return message;
  }
  return message;
}
