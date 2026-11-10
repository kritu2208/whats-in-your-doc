import { FormEvent, useEffect, useRef, useState } from "react";

import {
  Citation,
  ChatImage,
  DocumentItem,
  ChunkPreview,
  getChunkPreview,
  getApiUrl,
  listDocuments,
  streamQuestion,
  uploadDocument,
} from "./api";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  images?: ChatImage[];
  latencyMs?: number;
};
type SavedChat = {
  conversationId: string | null;
  messages: ChatMessage[];
};

const CHAT_STORAGE_KEY = "hybrid-rag.chat.v1";
const ACCEPTED_FILES = ".pdf,.md,.txt";
const CHAT_RETRIEVAL_MODE = "hybrid_rerank";

function readSavedChat(): SavedChat {
  try {
    const saved = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!saved) return { conversationId: null, messages: [] };
    const parsed: unknown = JSON.parse(saved);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "messages" in parsed &&
      Array.isArray(parsed.messages)
    ) {
      return {
        conversationId:
          "conversationId" in parsed && typeof parsed.conversationId === "string"
            ? parsed.conversationId
            : null,
        messages: parsed.messages as ChatMessage[],
      };
    }
  } catch {
    // A corrupt or unavailable local transcript should not prevent using the app.
  }
  return { conversationId: null, messages: [] };
}

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [chat, setChat] = useState<SavedChat>(readSavedChat);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(
    () => new Set(),
  );
  const [selectedChunk, setSelectedChunk] = useState<ChunkPreview | null>(null);
  const [isLoadingChunk, setIsLoadingChunk] = useState(false);
  const [question, setQuestion] = useState("");
  const [includeImages, setIncludeImages] = useState(false);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);
  const questionRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    void refreshDocuments();
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chat));
    } catch {
      setError("This browser could not save the conversation locally.");
    }
  }, [chat]);

  useEffect(() => {
    const list = messageListRef.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [chat.messages, isSending]);

  async function refreshDocuments(): Promise<void> {
    setIsLoadingDocuments(true);
    try {
      const result = await listDocuments();
      setDocuments(result);
      setSelectedDocumentIds((current) =>
        current.filter((id) => result.some((document) => document.id === id)),
      );
      setError(null);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not load documents.",
      );
    } finally {
      setIsLoadingDocuments(false);
    }
  }

  async function handleUpload(file: File | undefined): Promise<void> {
    if (!file) return;
    setError(null);
    setUploadNotice(null);
    setIsUploading(true);
    try {
      const created = await uploadDocument(file);
      setUploadNotice(`${created.filename} is ready to use.`);
      await refreshDocuments();
      setSelectedDocumentIds((current) =>
        current.includes(created.id) ? current : [...current, created.id],
      );
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The document could not be uploaded.",
      );
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function toggleDocument(documentId: string): void {
    setSelectedDocumentIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  }

  function toggleSources(messageId: string): void {
    setExpandedSources((current) => {
      const next = new Set(current);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      return next;
    });
  }

  async function openChunkPreview(citation: Citation): Promise<void> {
    setError(null);
    setIsLoadingChunk(true);
    try {
      setSelectedChunk(
        await getChunkPreview(citation.document_id, citation.chunk_id),
      );
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not load the cited source chunk.",
      );
    } finally {
      setIsLoadingChunk(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const submittedQuestion = question.trim();
    if (!submittedQuestion || isSending || isUploading) return;

    setIsSending(true);
    setError(null);
    setUploadNotice(null);
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: submittedQuestion,
    };
    const assistantId = crypto.randomUUID();
    setChat((current) => ({
      ...current,
      messages: [
        ...current.messages,
        userMessage,
        { id: assistantId, role: "assistant", content: "" },
      ],
    }));
    setQuestion("");

    try {
      const response = await streamQuestion({
        question: submittedQuestion,
        ...(chat.conversationId ? { conversation_id: chat.conversationId } : {}),
        document_ids: selectedDocumentIds.length ? selectedDocumentIds : null,
        retrieval_mode: CHAT_RETRIEVAL_MODE,
        top_k: 5,
        include_images: includeImages,
      }, (delta) => {
        setChat((current) => ({
          ...current,
          messages: current.messages.map((message) =>
            message.id === assistantId
              ? { ...message, content: message.content + delta }
              : message,
          ),
        }));
      }, (answer) => {
        setChat((current) => ({
          ...current,
          messages: current.messages.map((message) =>
            message.id === assistantId ? { ...message, content: answer } : message,
          ),
        }));
      });
      setChat((current) => ({
        conversationId: response.conversation_id,
        messages: current.messages.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: response.answer,
                citations: response.citations,
                images: response.images,
                latencyMs: response.latency_ms,
              }
            : message,
        ),
      }));
    } catch (requestError) {
      setChat((current) => ({
        ...current,
        messages: current.messages.filter(
          (message) => message.id !== userMessage.id && message.id !== assistantId,
        ),
      }));
      setQuestion(submittedQuestion);
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The question could not be answered.",
      );
    } finally {
      setIsSending(false);
      questionRef.current?.focus();
    }
  }

  function startNewConversation(): void {
    if (isSending) return;
    setChat({ conversationId: null, messages: [] });
    setExpandedSources(new Set());
    setQuestion("");
    setError(null);
    setUploadNotice(null);
    questionRef.current?.focus();
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="What's in the doc home">
          <span className="brand-mark" aria-hidden="true">
            ♡
          </span>
          <span className="brand-copy">
            <strong>What's in the doc</strong>
          </span>
        </a>
      </header>

      <div className="workspace">
        <aside className="sidebar" aria-label="Document library">
          <div className="sidebar-heading">
            <div>
              <span className="section-kicker">YOUR LIBRARY</span>
              <h2>Documents</h2>
            </div>
            <button
              className="icon-button"
              type="button"
              aria-label="Refresh documents"
              onClick={() => void refreshDocuments()}
              disabled={isLoadingDocuments}
            >
              ↻
            </button>
          </div>

          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            accept={ACCEPTED_FILES}
            onChange={(event) => void handleUpload(event.target.files?.[0])}
          />
          <button
            className="upload-button"
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            <span className="upload-plus" aria-hidden="true">
              {isUploading ? "…" : "+"}
            </span>
            {isUploading ? "Uploading document…" : "Upload a document"}
          </button>
          <p className="upload-help">PDF, Markdown or TXT · up to 10 MB</p>

          <div className="library-label">
            <span>AVAILABLE SOURCES</span>
            <span className="document-count">{documents.length}</span>
          </div>
          <div className="document-list">
            {isLoadingDocuments ? (
              <p className="muted-copy">Loading your library…</p>
            ) : documents.length === 0 ? (
              <div className="empty-library">
                <span aria-hidden="true">▤</span>
                <p>No documents yet</p>
                <small>Upload a file to start asking questions.</small>
              </div>
            ) : (
              documents.map((document) => (
                <label className="document-item" key={document.id}>
                  <input
                    type="checkbox"
                    checked={selectedDocumentIds.includes(document.id)}
                    onChange={() => toggleDocument(document.id)}
                  />
                  <span className={`file-icon file-icon--${fileType(document.filename)}`}>
                    {fileType(document.filename)}
                  </span>
                  <span className="document-copy">
                    <span className="document-name" title={document.filename}>
                      {document.filename}
                    </span>
                    <span className="document-meta">
                      {document.media_type} · {document.status}
                    </span>
                  </span>
                  <span
                    className={`document-state document-state--${document.status.toLowerCase()}`}
                    title={`Status: ${document.status}`}
                    aria-label={`Status: ${document.status}`}
                  />
                </label>
              ))
            )}
          </div>

          <div className="sidebar-bottom">
            <p className="selection-note">
              {selectedDocumentIds.length
                ? `${selectedDocumentIds.length} source${selectedDocumentIds.length === 1 ? "" : "s"} selected`
                : "Searching all available documents"}
            </p>
          </div>
        </aside>

        <section className="chat-panel" aria-label="Document question and answer">
          <div className="chat-heading">
            <div>
              <h1>Your documents, made easier to explore</h1>
            </div>
            <button
              className="new-chat-button"
              type="button"
              onClick={startNewConversation}
              disabled={isSending}
            >
              <span aria-hidden="true">＋</span> New chat
            </button>
          </div>

          {(error || uploadNotice) && (
            <div
              className={`notice ${error ? "notice--error" : "notice--success"}`}
              role={error ? "alert" : "status"}
            >
              <span>{error ?? uploadNotice}</span>
              {error && (
                <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
                  ×
                </button>
              )}
            </div>
          )}

          <div className="message-list" ref={messageListRef} aria-live="polite">
            {chat.messages.length === 0 ? (
              <div className="welcome-state">
                <div className="welcome-symbol" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
                <h2>Hey, what’s in your doc?</h2>
                <p>
                Ask me about your files and I’ll help you find the details.
                </p>
                <div className="prompt-suggestions">
                  <button
                    type="button"
                    onClick={() => setQuestion("Summarize the key points in my documents.")}
                    disabled={documents.length === 0}
                  >
                    Summarize a document
                  </button>
                  <button
                    type="button"
                    onClick={() => setQuestion("What are the main recommendations?")}
                    disabled={documents.length === 0}
                  >
                    Find a specific detail
                  </button>
                </div>
              </div>
            ) : (
              chat.messages.map((message) => (
                <article className={`message message--${message.role}`} key={message.id}>
                  <div className="message-avatar" aria-hidden="true">
                    {message.role === "user" ? "Y" : "♡"}
                  </div>
                  <div className="message-body">
                    <div className="message-label">
                      {message.role === "user" ? "You" : "What’s in the doc"}
                    </div>
                    <p className="message-content">{message.content}</p>
                    {message.role === "assistant" &&
                      ((message.citations?.length ?? 0) > 0 ||
                        (message.images?.length ?? 0) > 0) && (
                        <div className="source-section">
                          <button
                            className="sources-button"
                            type="button"
                            aria-expanded={expandedSources.has(message.id)}
                            onClick={() => toggleSources(message.id)}
                          >
                            <span aria-hidden="true">↗</span>
                            {expandedSources.has(message.id)
                              ? "Hide sources"
                              : `View sources (${(message.citations?.length ?? 0) + (message.images?.length ?? 0)})`}
                            <span className="sources-chevron" aria-hidden="true">
                              {expandedSources.has(message.id) ? "⌃" : "⌄"}
                            </span>
                          </button>
                          {expandedSources.has(message.id) && (
                            <div className="sources-content">
                              {message.images && message.images.length > 0 && (
                                <div className="retrieved-images">
                                  <div className="citation-heading">
                                    <span aria-hidden="true">▧</span> FIGURES
                                  </div>
                                  <div className="retrieved-image-grid">
                                    {message.images.map((image) => (
                                      <figure
                                        className="retrieved-image-card"
                                        key={`${message.id}-${image.image_id}`}
                                      >
                                        <a
                                          href={getApiUrl(image.image_url)}
                                          target="_blank"
                                          rel="noreferrer"
                                        >
                                          <img
                                            src={getApiUrl(image.image_url)}
                                            alt={`${image.document_name}, page ${image.page_number}`}
                                            loading="lazy"
                                          />
                                        </a>
                                        <figcaption>
                                          {image.document_name} · Page {image.page_number}
                                          {image.cited && " · Used for answer"}
                                        </figcaption>
                                        {image.context_excerpt && (
                                          <p className="image-context">
                                            Nearby text: {image.context_excerpt}
                                          </p>
                                        )}
                                      </figure>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {message.citations && message.citations.length > 0 && (
                                <div className="citations">
                                  {message.citations.map((citation) => (
                                    <button
                                      type="button"
                                      className="citation-card"
                                      key={`${message.id}-${citation.chunk_id}`}
                                      onClick={() => void openChunkPreview(citation)}
                                    >
                                      <div className="citation-title">
                                        <span className="citation-file" aria-hidden="true">
                                          ▤
                                        </span>
                                        <strong>{citation.document_name}</strong>
                                        {citation.page_number !== null && (
                                          <span className="citation-page">
                                            Page {citation.page_number}
                                          </span>
                                        )}
                                      </div>
                                      <p>“{citation.excerpt}”</p>
                                      <span className="chunk-reference">
                                        Chunk {citation.chunk_index + 1} · {citation.chunk_id.slice(0, 8)}
                                      </span>
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    {message.role === "assistant" && message.latencyMs !== undefined && (
                      <span className="response-time">
                        Answered in {(message.latencyMs / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>
                </article>
              ))
            )}
            {isSending && (
              <div className="message message--assistant" role="status" aria-label="Answering">
                <div className="message-avatar" aria-hidden="true">
                  ♡
                </div>
                <div className="message-body">
                  <div className="message-label">What’s in the doc</div>
                  <div className="typing-indicator">
                    <span />
                    <span />
                    <span />
                    <span className="typing-label">Streaming a grounded answer…</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <form className="composer" onSubmit={(event) => void handleSubmit(event)}>
            <label className="visually-hidden" htmlFor="question">
              Ask a question about your documents
            </label>
            <textarea
              id="question"
              ref={questionRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder={
                documents.length
                ? "Ask about your document…"
                : "Upload a file to begin…"
              }
              rows={2}
              maxLength={10_000}
              disabled={isSending || isUploading}
            />
            <div className="composer-footer">
              <span>Enter to send · Shift + Enter for a new line</span>
              <div className="composer-actions">
                <button
                  className={`image-toggle${includeImages ? " image-toggle--active" : ""}`}
                  type="button"
                  aria-pressed={includeImages}
                  title="Show extracted PDF images from pages with retrieved text"
                  onClick={() => setIncludeImages((enabled) => !enabled)}
                  disabled={isSending || isUploading}
                >
                  <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
                    <rect x="2.5" y="3.5" width="15" height="13" rx="2" />
                    <circle cx="7" cy="8" r="1.5" />
                    <path d="m4 14 4-3 2.5 2 2.5-3 3 4" />
                  </svg>
                  PDF images <span>{includeImages ? "On" : "Off"}</span>
                </button>
                <button
                  className="send-button"
                  type="submit"
                  disabled={!question.trim() || isSending || isUploading}
                >
                  {isSending ? "Thinking…" : "Ask"}
                  <span aria-hidden="true">↗</span>
                </button>
              </div>
            </div>
          </form>
        </section>
      </div>
      {(selectedChunk || isLoadingChunk) && (
        <div
          className="source-modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setSelectedChunk(null);
          }}
        >
          <section
            className="source-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="source-modal-title"
          >
            <button
              className="source-modal-close"
              type="button"
              aria-label="Close source preview"
              onClick={() => setSelectedChunk(null)}
            >
              ×
            </button>
            {isLoadingChunk ? (
              <p>Loading source preview…</p>
            ) : selectedChunk ? (
              <>
                <p className="citation-heading">CITED SOURCE</p>
                <h2 id="source-modal-title">{selectedChunk.document_name}</h2>
                <p className="source-modal-meta">
                  {selectedChunk.page_number !== null
                    ? `Page ${selectedChunk.page_number} · `
                    : ""}
                  Chunk {selectedChunk.chunk_index + 1}
                </p>
                <pre>{selectedChunk.content}</pre>
              </>
            ) : null}
          </section>
        </div>
      )}
    </main>
  );
}

function fileType(filename: string): string {
  const extension = filename.split(".").pop()?.toLowerCase();
  if (extension === "pdf") return "PDF";
  if (extension === "md" || extension === "markdown") return "MD";
  return "TXT";
}

export default App;
