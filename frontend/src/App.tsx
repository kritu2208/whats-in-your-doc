import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  Citation,
  ChatImage,
  ConversationSummary,
  DocumentItem,
  ChunkPreview,
  deleteConversation,
  deleteDocument,
  getConversation,
  getChunkPreview,
  getApiUrl,
  listConversations,
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
type ThemeMode = "pink" | "bright" | "dark";
type NoteItem = {
  id: string;
  title: string;
  content: string;
  updatedAt: string;
};

type SidebarTab = "documents" | "chats";
type CachedMessageExtras = {
  images?: ChatImage[];
  latencyMs?: number;
};

const CHAT_STORAGE_KEY = "hybrid-rag.chat.v1";
const CHAT_MEDIA_STORAGE_KEY = "hybrid-rag.chat-media.v1";
const THEME_STORAGE_KEY = "hybrid-rag.theme.v1";
const NOTES_STORAGE_KEY = "hybrid-rag.notes.v1";
const ACCEPTED_FILES = ".pdf,.md,.txt";
const CHAT_RETRIEVAL_MODE = "hybrid_rerank";

function readChatMediaCache(): Record<string, CachedMessageExtras[]> {
  try {
    const raw = localStorage.getItem(CHAT_MEDIA_STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as Record<string, CachedMessageExtras[]>;
    }
  } catch {
    // Ignore invalid local media cache.
  }
  return {};
}

function saveConversationExtras(
  conversationId: string,
  messages: ChatMessage[],
): void {
  try {
    const cache = readChatMediaCache();
    cache[conversationId] = messages
      .filter((msg) => msg.role === "assistant")
      .map((msg) => ({
        images: msg.images,
        latencyMs: msg.latencyMs,
      }));
    localStorage.setItem(CHAT_MEDIA_STORAGE_KEY, JSON.stringify(cache));
  } catch {
    // Best-effort cache for inline images/latency on revisited chats.
  }
}

function removeConversationExtras(conversationId: string): void {
  try {
    const cache = readChatMediaCache();
    if (conversationId in cache) {
      delete cache[conversationId];
      localStorage.setItem(CHAT_MEDIA_STORAGE_KEY, JSON.stringify(cache));
    }
  } catch {
    // Ignore storage cleanup errors.
  }
}

function RibbonBowIcon({ className = "cute-icon" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M12 11.2c-1.8-3.4-5.4-5.4-7.4-4-1.8 1.2-1.5 4.9.9 6.2 1.9 1 4.8.4 6.5-2.2Zm0 0c1.8-3.4 5.4-5.4 7.4-4 1.8 1.2 1.5 4.9-.9 6.2-1.9 1-4.8.4-6.5-2.2Z"
        fill="currentColor"
        fillOpacity="0.24"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <path
        d="M10.4 13.4 7.5 19.2l3-1.4 1.5 1.8m1.6-6.2 2.9 5.8-3-1.4-1.5 1.8"
        stroke="currentColor"
        strokeWidth="1.65"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="11.8" r="2.1" fill="currentColor" />
    </svg>
  );
}

function FlowerIcon({ className = "cute-icon" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="12" cy="5.8" r="3" fill="currentColor" fillOpacity="0.32" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="17.8" cy="10" r="3" fill="currentColor" fillOpacity="0.32" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="15.6" cy="16.8" r="3" fill="currentColor" fillOpacity="0.32" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="8.4" cy="16.8" r="3" fill="currentColor" fillOpacity="0.32" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="6.2" cy="10" r="3" fill="currentColor" fillOpacity="0.32" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="12" cy="11.9" r="2.5" fill="currentColor" />
    </svg>
  );
}

function TrashIcon({ className = "cute-icon" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4.5 5.5h11m-9-2h7m-7.5 2 .7 9.6a1.5 1.5 0 0 0 1.5 1.4h3.6a1.5 1.5 0 0 0 1.5-1.4l.7-9.6M8.5 8.5v5m3-5v5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function readSavedTheme(): ThemeMode {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "pink" || saved === "bright" || saved === "dark") {
      return saved;
    }
  } catch {
    // Default back to the signature pink theme if storage is restricted.
  }
  return "pink";
}

function readSavedNotes(): NoteItem[] {
  try {
    const saved = localStorage.getItem(NOTES_STORAGE_KEY);
    if (!saved) return [];
    const parsed: unknown = JSON.parse(saved);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is NoteItem =>
        typeof item === "object" &&
        item !== null &&
        typeof (item as NoteItem).id === "string" &&
        typeof (item as NoteItem).title === "string" &&
        typeof (item as NoteItem).content === "string" &&
        typeof (item as NoteItem).updatedAt === "string",
    );
  } catch {
    return [];
  }
}

function formatNoteDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function selectAnswerImages(images?: ChatImage[], content?: string): ChatImage[] {
  if (!images || images.length === 0) return [];
  if (
    content &&
    (content.toLowerCase().includes("do not provide enough information") ||
      content.toLowerCase().includes("does not provide enough information") ||
      content.toLowerCase().includes("cannot answer"))
  ) {
    return [];
  }
  return images.filter((image) => image.cited);
}

function FigureCard({
  image,
  showContext = true,
}: {
  image: ChatImage;
  showContext?: boolean;
}) {
  return (
    <figure className="retrieved-image-card">
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
      {showContext && image.context_excerpt && (
        <p className="image-context">
          Nearby text: {image.context_excerpt}
        </p>
      )}
    </figure>
  );
}

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
  const [isDeleting, setIsDeleting] = useState(false);
  const [pendingDeleteIds, setPendingDeleteIds] = useState<string[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const [theme, setTheme] = useState<ThemeMode>(readSavedTheme);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("documents");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [selectedChatIds, setSelectedChatIds] = useState<string[]>([]);
  const [pendingDeleteChatIds, setPendingDeleteChatIds] = useState<string[]>([]);
  const [isDeletingChat, setIsDeletingChat] = useState(false);
  const [isNotesOpen, setIsNotesOpen] = useState(false);
  const [notes, setNotes] = useState<NoteItem[]>(readSavedNotes);
  const [noteTitle, setNoteTitle] = useState("");
  const [noteContent, setNoteContent] = useState("");
  const [editingNoteId, setEditingNoteId] = useState<string | null>(null);
  const [savedMessageIds, setSavedMessageIds] = useState<Set<string>>(
    () => new Set(),
  );
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messageListRef = useRef<HTMLDivElement>(null);
  const questionRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    void refreshDocuments();
    void refreshConversations();
  }, []);

  useEffect(() => {
    document.body.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Ignore storage failures for UI preference.
    }
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem(NOTES_STORAGE_KEY, JSON.stringify(notes));
    } catch {
      setError("This browser could not save your notes locally.");
    }
  }, [notes]);

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

  async function refreshConversations(): Promise<void> {
    setIsLoadingConversations(true);
    try {
      const result = await listConversations();
      setConversations(result);
      setSelectedChatIds((current) =>
        current.filter((id) => result.some((item) => item.id === id)),
      );
    } catch {
      // Keep chat UI usable if conversation history list temporarily fails.
    } finally {
      setIsLoadingConversations(false);
    }
  }

  async function openSavedConversation(conversationId: string): Promise<void> {
    if (isSending || isLoadingConversation) return;
    setError(null);
    setUploadNotice(null);
    setPendingDeleteChatIds([]);
    setIsLoadingConversation(true);
    try {
      const detail = await getConversation(conversationId);
      const cachedExtras = readChatMediaCache()[conversationId] ?? [];
      let assistantIndex = 0;
      const mappedMessages: ChatMessage[] = detail.messages.map((item) => {
        if (item.role === "assistant") {
          const extras = cachedExtras[assistantIndex];
          assistantIndex += 1;
          return {
            id: item.id,
            role: "assistant",
            content: item.content,
            citations: item.citations,
            images: extras?.images,
            latencyMs: extras?.latencyMs,
          };
        }
        return {
          id: item.id,
          role: "user",
          content: item.content,
        };
      });
      setChat({
        conversationId: detail.id,
        messages: mappedMessages,
      });
      setExpandedSources(new Set());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not load that saved chat.",
      );
    } finally {
      setIsLoadingConversation(false);
    }
  }

  function toggleSelectAllChats(): void {
    setPendingDeleteChatIds([]);
    if (allChatsSelected) {
      setSelectedChatIds([]);
    } else {
      setSelectedChatIds(conversations.map((item) => item.id));
    }
  }

  function toggleChatSelect(conversationId: string): void {
    setPendingDeleteChatIds([]);
    setSelectedChatIds((current) =>
      current.includes(conversationId)
        ? current.filter((id) => id !== conversationId)
        : [...current, conversationId],
    );
  }

  function requestDeleteChats(ids: string[]): void {
    if (isDeletingChat || ids.length === 0) return;
    setError(null);
    setUploadNotice(null);
    setPendingDeleteChatIds(ids);
  }

  async function confirmDeleteChats(): Promise<void> {
    if (isDeletingChat || pendingDeleteChatIds.length === 0) return;
    const idsToDelete = [...pendingDeleteChatIds];
    setIsDeletingChat(true);
    setError(null);
    setUploadNotice(null);
    try {
      const results = await Promise.allSettled(
        idsToDelete.map((id) => deleteConversation(id)),
      );
      const deletedIds = idsToDelete.filter(
        (_, index) => results[index]?.status === "fulfilled",
      );
      const failedCount = idsToDelete.length - deletedIds.length;
      if (deletedIds.length > 0) {
        deletedIds.forEach((id) => removeConversationExtras(id));
        setConversations((current) =>
          current.filter((item) => !deletedIds.includes(item.id)),
        );
        setSelectedChatIds((current) =>
          current.filter((id) => !deletedIds.includes(id)),
        );
        if (chat.conversationId && deletedIds.includes(chat.conversationId)) {
          setChat({ conversationId: null, messages: [] });
          setExpandedSources(new Set());
        }
        setUploadNotice(
          deletedIds.length === 1
            ? "Deleted 1 saved chat."
            : `Deleted ${deletedIds.length} saved chats.`,
        );
      }
      if (failedCount > 0) {
        setError(
          `Could not delete ${failedCount} saved chat${failedCount === 1 ? "" : "s"}.`,
        );
      }
      setPendingDeleteChatIds([]);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not delete the conversations.",
      );
    } finally {
      setIsDeletingChat(false);
    }
  }

  async function handleUpload(file: File | undefined): Promise<void> {
    if (!file) return;
    setError(null);
    setUploadNotice(null);
    const duplicateByName = documents.find(
      (document) => document.filename.toLowerCase() === file.name.toLowerCase(),
    );
    if (duplicateByName) {
      setError(`Can't upload '${file.name}': it already exists.`);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
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
    setPendingDeleteIds([]);
    setSelectedDocumentIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  }

  function toggleSelectAllDocuments(): void {
    setPendingDeleteIds([]);
    if (selectedDocumentIds.length === documents.length) {
      setSelectedDocumentIds([]);
    } else {
      setSelectedDocumentIds(documents.map((document) => document.id));
    }
  }

  function requestDeleteDocuments(ids: string[]): void {
    if (isDeleting || ids.length === 0) return;
    setError(null);
    setUploadNotice(null);
    setPendingDeleteIds(ids);
  }

  async function confirmDeleteDocuments(): Promise<void> {
    if (isDeleting || pendingDeleteIds.length === 0) return;
    const idsToDelete = [...pendingDeleteIds];
    setIsDeleting(true);
    setError(null);
    setUploadNotice(null);
    try {
      const results = await Promise.allSettled(
        idsToDelete.map((id) => deleteDocument(id)),
      );
      const deletedIds = idsToDelete.filter(
        (_, index) => results[index]?.status === "fulfilled",
      );
      const failedCount = idsToDelete.length - deletedIds.length;
      if (deletedIds.length > 0) {
        setSelectedDocumentIds((current) =>
          current.filter((id) => !deletedIds.includes(id)),
        );
        setUploadNotice(
          deletedIds.length === 1
            ? "Deleted 1 document."
            : `Deleted ${deletedIds.length} documents.`,
        );
      }
      if (failedCount > 0) {
        setError(
          `Could not delete ${failedCount} document${failedCount === 1 ? "" : "s"}.`,
        );
      }
      setPendingDeleteIds([]);
      await refreshDocuments();
    } finally {
      setIsDeleting(false);
    }
  }

  function handleSaveNote(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const trimmedContent = noteContent.trim();
    if (!trimmedContent) return;
    const trimmedTitle = noteTitle.trim() || "Quick study note";
    const now = new Date().toISOString();

    if (editingNoteId) {
      setNotes((current) =>
        current.map((note) =>
          note.id === editingNoteId
            ? { ...note, title: trimmedTitle, content: trimmedContent, updatedAt: now }
            : note,
        ),
      );
      setEditingNoteId(null);
    } else {
      setNotes((current) => [
        {
          id: crypto.randomUUID(),
          title: trimmedTitle,
          content: trimmedContent,
          updatedAt: now,
        },
        ...current,
      ]);
    }
    setNoteTitle("");
    setNoteContent("");
  }

  function startEditingNote(note: NoteItem): void {
    setEditingNoteId(note.id);
    setNoteTitle(note.title);
    setNoteContent(note.content);
    setIsNotesOpen(true);
  }

  function handleDeleteNote(noteId: string): void {
    setNotes((current) => current.filter((note) => note.id !== noteId));
    if (editingNoteId === noteId) {
      setEditingNoteId(null);
      setNoteTitle("");
      setNoteContent("");
    }
  }

  function saveMessageToNotes(message: ChatMessage, index: number): void {
    const content = message.content.trim();
    if (!content) return;
    const previousUser = [...chat.messages.slice(0, index)]
      .reverse()
      .find((entry) => entry.role === "user");
    const derivedTitle = previousUser?.content.trim()
      ? previousUser.content.trim().slice(0, 56)
      : "Saved document note";

    setNotes((current) => [
      {
        id: crypto.randomUUID(),
        title: derivedTitle,
        content,
        updatedAt: new Date().toISOString(),
      },
      ...current,
    ]);
    setSavedMessageIds((current) => new Set(current).add(message.id));
    setIsNotesOpen(true);
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
      setChat((current) => {
        const nextMessages = current.messages.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: response.answer,
                citations: response.citations,
                images: response.images,
                latencyMs: response.latency_ms,
              }
            : message,
        );
        saveConversationExtras(response.conversation_id, nextMessages);
        return {
          conversationId: response.conversation_id,
          messages: nextMessages,
        };
      });
      void refreshConversations();
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

  const allSelected =
    documents.length > 0 && selectedDocumentIds.length === documents.length;
  const allChatsSelected =
    conversations.length > 0 && selectedChatIds.length === conversations.length;

  return (
    <main className="app-shell" data-theme={theme}>
      <header className="topbar">
        <a className="brand" href="/" aria-label="What's in the doc home">
          <span className="brand-mark" aria-hidden="true">
            ♡
          </span>
          <span className="brand-copy">
            <strong>What's in the doc</strong>
          </span>
        </a>
        <div className="theme-switcher" role="group" aria-label="Color theme">
          {(
            [
              { id: "pink", label: "Pink" },
              { id: "bright", label: "Bright" },
              { id: "dark", label: "Dark" },
            ] as const
          ).map((option) => (
            <button
              key={option.id}
              type="button"
              className={`theme-pill${theme === option.id ? " theme-pill--active" : ""}`}
              aria-pressed={theme === option.id}
              onClick={() => setTheme(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </header>

      <div className={`workspace${isNotesOpen ? " workspace--notes-open" : ""}`}>
        <aside className="sidebar" aria-label="Workspace sidebar">
          <div className="sidebar-tabs" role="tablist" aria-label="Sidebar view">
            <button
              type="button"
              role="tab"
              aria-selected={sidebarTab === "documents"}
              className={`sidebar-tab${sidebarTab === "documents" ? " sidebar-tab--active" : ""}`}
              onClick={() => setSidebarTab("documents")}
            >
              Documents
              <span className="sidebar-tab-count">{documents.length}</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={sidebarTab === "chats"}
              className={`sidebar-tab${sidebarTab === "chats" ? " sidebar-tab--active" : ""}`}
              onClick={() => {
                setSidebarTab("chats");
                void refreshConversations();
              }}
            >
              Saved Chats
              <span className="sidebar-tab-count">{conversations.length}</span>
            </button>
          </div>

          {sidebarTab === "documents" ? (
            <>
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
                  disabled={isLoadingDocuments || isDeleting}
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
                disabled={isUploading || isDeleting}
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

              {documents.length > 0 && (
                <div className="library-actions">
                  <button
                    className="library-action-btn"
                    type="button"
                    onClick={toggleSelectAllDocuments}
                    disabled={isDeleting}
                  >
                    {allSelected ? "Unselect all" : "Select all"}
                  </button>
                  <button
                    className="library-action-btn library-action-btn--danger"
                    type="button"
                    onClick={() => requestDeleteDocuments(selectedDocumentIds)}
                    disabled={selectedDocumentIds.length === 0 || isDeleting}
                  >
                    <TrashIcon className="action-trash-icon" />
                    Delete{selectedDocumentIds.length > 0 ? ` (${selectedDocumentIds.length})` : ""}
                  </button>
                </div>
              )}

              {pendingDeleteIds.length > 0 && (
                <div className="delete-confirm-box" role="alert">
                  <p>
                    Delete{" "}
                    {pendingDeleteIds.length === 1
                      ? `"${
                          documents.find((doc) => doc.id === pendingDeleteIds[0])
                            ?.filename ?? "selected document"
                        }"`
                      : `${pendingDeleteIds.length} selected documents`}
                    ?
                  </p>
                  <div className="delete-confirm-actions">
                    <button
                      className="delete-confirm-btn"
                      type="button"
                      onClick={() => void confirmDeleteDocuments()}
                      disabled={isDeleting}
                    >
                      {isDeleting ? "Deleting…" : "Yes, delete"}
                    </button>
                    <button
                      className="delete-cancel-btn"
                      type="button"
                      onClick={() => setPendingDeleteIds([])}
                      disabled={isDeleting}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

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
                    <div className="document-row" key={document.id}>
                      <label className="document-item">
                        <input
                          type="checkbox"
                          checked={selectedDocumentIds.includes(document.id)}
                          onChange={() => toggleDocument(document.id)}
                          disabled={isDeleting}
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
                      <button
                        className="document-delete-button"
                        type="button"
                        title={`Delete ${document.filename}`}
                        aria-label={`Delete ${document.filename}`}
                        onClick={() => requestDeleteDocuments([document.id])}
                        disabled={isDeleting}
                      >
                        <TrashIcon className="row-trash-icon" />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </>
          ) : (
            <>
              <div className="sidebar-heading">
                <div>
                  <span className="section-kicker">HISTORY</span>
                  <h2>Saved Chats</h2>
                </div>
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Refresh saved chats"
                  onClick={() => void refreshConversations()}
                  disabled={isLoadingConversations || isDeletingChat}
                >
                  ↻
                </button>
              </div>

              <button
                className="upload-button"
                type="button"
                onClick={startNewConversation}
                disabled={isSending}
              >
                <span className="upload-plus" aria-hidden="true">
                  +
                </span>
                Start a new chat
              </button>
              <p className="upload-help">Click any saved chat to resume it anytime</p>

              <div className="library-label">
                <span>RECENT CONVERSATIONS</span>
                <span className="document-count">{conversations.length}</span>
              </div>

              {conversations.length > 0 && (
                <div className="library-actions">
                  <button
                    className="library-action-btn"
                    type="button"
                    onClick={toggleSelectAllChats}
                    disabled={isDeletingChat}
                  >
                    {allChatsSelected ? "Unselect all" : "Select all"}
                  </button>
                  <button
                    className="library-action-btn library-action-btn--danger"
                    type="button"
                    onClick={() => requestDeleteChats(selectedChatIds)}
                    disabled={selectedChatIds.length === 0 || isDeletingChat}
                  >
                    <TrashIcon className="action-trash-icon" />
                    Delete{selectedChatIds.length > 0 ? ` (${selectedChatIds.length})` : ""}
                  </button>
                </div>
              )}

              {pendingDeleteChatIds.length > 0 && (
                <div className="delete-confirm-box" role="alert">
                  <p>
                    Delete{" "}
                    {pendingDeleteChatIds.length === 1
                      ? `"${
                          conversations.find((item) => item.id === pendingDeleteChatIds[0])
                            ?.title ?? "selected chat"
                        }"`
                      : `${pendingDeleteChatIds.length} selected chats`}
                    ?
                  </p>
                  <div className="delete-confirm-actions">
                    <button
                      className="delete-confirm-btn"
                      type="button"
                      onClick={() => void confirmDeleteChats()}
                      disabled={isDeletingChat}
                    >
                      {isDeletingChat ? "Deleting…" : "Yes, delete"}
                    </button>
                    <button
                      className="delete-cancel-btn"
                      type="button"
                      onClick={() => setPendingDeleteChatIds([])}
                      disabled={isDeletingChat}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              <div className="chat-history-list">
                {isLoadingConversations ? (
                  <p className="muted-copy">Loading saved chats…</p>
                ) : conversations.length === 0 ? (
                  <div className="empty-library">
                    <span aria-hidden="true">♡</span>
                    <p>No saved chats yet</p>
                    <small>Ask a question and your conversation will appear here.</small>
                  </div>
                ) : (
                  conversations.map((item) => {
                    const isActive = chat.conversationId === item.id;
                    const isSelected = selectedChatIds.includes(item.id);
                    return (
                      <div
                        className={`chat-history-row${isActive ? " chat-history-row--active" : ""}`}
                        key={item.id}
                      >
                        <input
                          type="checkbox"
                          className="chat-history-checkbox"
                          checked={isSelected}
                          onChange={() => toggleChatSelect(item.id)}
                          disabled={isDeletingChat}
                          aria-label={`Select chat "${item.title}"`}
                        />
                        <button
                          type="button"
                          className="chat-history-item"
                          onClick={() => void openSavedConversation(item.id)}
                          disabled={isSending || isLoadingConversation}
                        >
                          <strong className="chat-history-title" title={item.title}>
                            {item.title}
                          </strong>
                          {item.preview && (
                            <span className="chat-history-preview">{item.preview}</span>
                          )}
                          <span className="chat-history-meta">
                            {item.message_count} msg{item.message_count === 1 ? "" : "s"} ·{" "}
                            {formatNoteDate(item.updated_at)}
                          </span>
                        </button>
                        <button
                          className="document-delete-button"
                          type="button"
                          title={`Delete chat "${item.title}"`}
                          aria-label={`Delete chat "${item.title}"`}
                          onClick={() => requestDeleteChats([item.id])}
                          disabled={isDeletingChat}
                        >
                          <TrashIcon className="row-trash-icon" />
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </>
          )}
        </aside>

        <section className="chat-panel" aria-label="Document question and answer">
          <div className="chat-heading">
            <div>
              <h1>Your documents, made easier to explore</h1>
            </div>
            <div className="chat-heading-actions">
              <button
                className={`notes-toggle-button${isNotesOpen ? " notes-toggle-button--active" : ""}`}
                type="button"
                aria-expanded={isNotesOpen}
                onClick={() => setIsNotesOpen((open) => !open)}
              >
                <RibbonBowIcon className="notes-bow-icon" />
                <span>Notes</span>
                {notes.length > 0 && (
                  <span className="notes-count-pill">{notes.length}</span>
                )}
              </button>
              <button
                className="new-chat-button"
                type="button"
                onClick={startNewConversation}
                disabled={isSending}
              >
                <span aria-hidden="true">＋</span> New chat
              </button>
            </div>
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
              chat.messages.map((message, index) => (
                <article className={`message message--${message.role}`} key={message.id}>
                  <div className="message-avatar" aria-hidden="true">
                    {message.role === "user" ? "Y" : "♡"}
                  </div>
                  <div className="message-body">
                    <div className="message-label">
                      {message.role === "user" ? "You" : "What’s in the doc"}
                    </div>
                    {message.role === "assistant" ? (
                      <div className="message-content message-content--markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="message-content">
                        <p>{message.content}</p>
                      </div>
                    )}
                    {message.role === "assistant" &&
                      selectAnswerImages(message.images, message.content).length > 0 && (
                        <div className="answer-images">
                          <div className="retrieved-image-grid">
                            {selectAnswerImages(message.images, message.content).map((image) => (
                              <FigureCard
                                key={`${message.id}-inline-${image.image_id}`}
                                image={image}
                                showContext={false}
                              />
                            ))}
                          </div>
                        </div>
                      )}
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
                                      <FigureCard
                                        key={`${message.id}-${image.image_id}`}
                                        image={image}
                                      />
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
                    {message.role === "assistant" && message.content.trim().length > 0 && (
                      <div className="message-meta-row">
                        {message.latencyMs !== undefined && (
                          <span className="response-time">
                            Answered in {(message.latencyMs / 1000).toFixed(1)}s
                          </span>
                        )}
                        <button
                          className={`save-note-pill${savedMessageIds.has(message.id) ? " save-note-pill--saved" : ""}`}
                          type="button"
                          onClick={() => saveMessageToNotes(message, index)}
                        >
                          <FlowerIcon className="note-pill-flower" />
                          {savedMessageIds.has(message.id) ? "Saved to notes" : "Save to notes"}
                        </button>
                      </div>
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

        {isNotesOpen && (
          <aside className="notes-panel" aria-label="Sweet study notes">
            <div className="notes-garland" aria-hidden="true">
              <FlowerIcon className="notes-garland-flower" />
              <span className="notes-garland-dot" />
              <RibbonBowIcon className="notes-garland-bow" />
              <span className="notes-garland-dot" />
              <FlowerIcon className="notes-garland-flower" />
            </div>

            <div className="notes-header">
              <div className="notes-title-group">
                <span className="notes-bow-badge" aria-hidden="true">
                  <RibbonBowIcon className="notes-header-bow" />
                </span>
                <div>
                  <span className="notes-kicker">BLOOM & RIBBON</span>
                  <h2>Sweet Notes</h2>
                </div>
              </div>
              <button
                className="notes-close-btn"
                type="button"
                aria-label="Close notes"
                onClick={() => setIsNotesOpen(false)}
              >
                ×
              </button>
            </div>

            <form className="notes-composer" onSubmit={handleSaveNote}>
              <input
                type="text"
                className="notes-title-input"
                placeholder="Note title (optional)…"
                value={noteTitle}
                onChange={(event) => setNoteTitle(event.target.value)}
                maxLength={120}
              />
              <textarea
                className="notes-textarea"
                placeholder="Jot down highlights, key quotes, or study reminders…"
                value={noteContent}
                onChange={(event) => setNoteContent(event.target.value)}
                rows={3}
                maxLength={4000}
              />
              <div className="notes-composer-actions">
                {editingNoteId && (
                  <button
                    type="button"
                    className="notes-cancel-edit"
                    onClick={() => {
                      setEditingNoteId(null);
                      setNoteTitle("");
                      setNoteContent("");
                    }}
                  >
                    Cancel edit
                  </button>
                )}
                <button
                  type="submit"
                  className="notes-save-btn"
                  disabled={!noteContent.trim()}
                >
                  <FlowerIcon className="notes-btn-flower" />
                  {editingNoteId ? "Update note" : "Save note"}
                </button>
              </div>
            </form>

            <div className="notes-list">
              {notes.length === 0 ? (
                <div className="notes-empty">
                  <div className="notes-empty-emblem" aria-hidden="true">
                    <FlowerIcon className="notes-empty-flower" />
                    <RibbonBowIcon className="notes-empty-bow" />
                  </div>
                  <p>No notes saved yet</p>
                  <small>
                    Write a note above or click <strong>Save to notes</strong> on any answer.
                  </small>
                </div>
              ) : (
                notes.map((note) => (
                  <article className="note-card" key={note.id}>
                    <div className="note-card-top">
                      <span className="note-card-bow" aria-hidden="true">
                        <RibbonBowIcon />
                      </span>
                      <strong className="note-card-title">{note.title}</strong>
                      <div className="note-card-actions">
                        <button
                          type="button"
                          className="note-card-action"
                          aria-label={`Edit note ${note.title}`}
                          onClick={() => startEditingNote(note)}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          className="note-card-action note-card-action--delete"
                          aria-label={`Delete note ${note.title}`}
                          onClick={() => handleDeleteNote(note.id)}
                        >
                          ×
                        </button>
                      </div>
                    </div>
                    <p className="note-card-body">{note.content}</p>
                    <div className="note-card-footer">
                      <FlowerIcon className="note-card-flower" />
                      <span>{formatNoteDate(note.updatedAt)}</span>
                    </div>
                  </article>
                ))
              )}
            </div>
          </aside>
        )}
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
