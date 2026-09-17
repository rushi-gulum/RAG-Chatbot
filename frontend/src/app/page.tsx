"use client";

import {
  ArrowUp, BookOpen, Check, ChevronDown, Copy, CopyCheck, FileText,
  LoaderCircle, LogOut, Menu, MessageSquare, MoreHorizontal,
  Paperclip, PanelRightOpen, Pencil, Plus, Search, ShieldCheck,
  Sparkles, Trash2, UploadCloud, Download, X, List,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { GoogleAuthProvider, onAuthStateChanged, signInWithPopup, signOut, User } from "firebase/auth";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { firebaseAuth } from "@/lib/firebase";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const LOCAL_AUTH_MODE = process.env.NEXT_PUBLIC_AUTH_MODE === "local";

type DocumentItem = { file_id: string; filename: string; uploaded_at: number; size_bytes: number };
type Source = { filename?: string; chunk_index?: number; similarity_score?: number };
type Message = { id: string; type: "user" | "assistant"; content: string; sources?: Source[] };
type ChatSession = { session_id: string; title: string; message_count: number; updated_at: string };

async function apiRequest(path: string, options: RequestInit = {}, user?: User | null) {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  if (user) headers.set("Authorization", `Bearer ${await user.getIdToken()}`);
  const response = await fetch(`${API_URL}${path}`, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "The request could not be completed.");
  return body;
}

function formatBytes(bytes: number) {
  return bytes ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : "PDF document";
}

function cleanAssistantResponse(content: string) {
  return content.replace(/^\s*\*\*Direct Answer\*\*\s*:?[ \t]*/i, "");
}

// ── Conversation Navigator Rail ─────────────────────────────────────────────
function ConversationRail({
  messages,
  onJump,
}: {
  messages: Message[];
  onJump: (id: string) => void;
}) {
  const [railOpen, setRailOpen] = useState(false);
  const userMessages = messages.filter((m) => m.type === "user");
  if (userMessages.length === 0) return null;

  return (
    <>
      {/* Desktop: thin tick rail pinned to right edge of chat area */}
      <nav className="conv-rail" aria-label="Jump to question">
        {userMessages.map((msg, i) => (
          <button
            key={msg.id}
            className="conv-rail-tick"
            onClick={() => onJump(msg.id)}
            aria-label={`Jump to question ${i + 1}: ${msg.content.slice(0, 60)}`}
          >
            <span className="conv-rail-tooltip">
              <span className="conv-rail-tooltip-num">Q{i + 1}</span>
              {msg.content.length > 48
                ? msg.content.slice(0, 48) + "…"
                : msg.content}
            </span>
          </button>
        ))}
      </nav>

      {/* Mobile: floating button + slide-up drawer */}
      <div className="conv-rail-mobile">
        <button
          className="conv-rail-mobile-trigger"
          onClick={() => setRailOpen((v) => !v)}
          aria-label="Jump to question"
          aria-expanded={railOpen}
        >
          <List size={16} />
          <span className="conv-rail-mobile-count">{userMessages.length}</span>
        </button>

        {railOpen && (
          <>
            <div
              className="conv-rail-mobile-backdrop"
              onClick={() => setRailOpen(false)}
            />
            <div className="conv-rail-mobile-drawer">
              <div className="conv-rail-mobile-header">
                <span>Questions in this chat</span>
                <button
                  className="conv-rail-mobile-close"
                  onClick={() => setRailOpen(false)}
                  aria-label="Close navigator"
                >
                  <X size={15} />
                </button>
              </div>
              <div className="conv-rail-mobile-list">
                {userMessages.map((msg, i) => (
                  <button
                    key={msg.id}
                    className="conv-rail-mobile-item"
                    onClick={() => {
                      onJump(msg.id);
                      setRailOpen(false);
                    }}
                  >
                    <span className="conv-rail-mobile-item-num">Q{i + 1}</span>
                    <span className="conv-rail-mobile-item-text">
                      {msg.content.length > 72
                        ? msg.content.slice(0, 72) + "…"
                        : msg.content}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStep, setUploadStep] = useState<"reading" | "uploading" | "indexing" | "done" | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [authInitialized, setAuthInitialized] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [showBackdrop, setShowBackdrop] = useState(false);
  const [sessionSearch, setSessionSearch] = useState("");
  const [showAllSessions, setShowAllSessions] = useState(false);
  const [activeMenu, setActiveMenu] = useState<string | null>(null);      // session_id of open context menu
  const [renamingId, setRenamingId] = useState<string | null>(null);       // session_id being renamed
  const [renameValue, setRenameValue] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null); // session_id pending delete confirm
  const [copiedId, setCopiedId] = useState<string | null>(null);              // message_id that was just copied
  const fileInput = useRef<HTMLInputElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const messageRefsMap = useRef<Map<string, HTMLElement>>(new Map());
  const conversationScrollRef = useRef<HTMLDivElement>(null);

  const setMessageRef = useCallback((id: string, el: HTMLElement | null) => {
    if (el) messageRefsMap.current.set(id, el);
    else messageRefsMap.current.delete(id);
  }, []);

  const jumpToMessage = useCallback((id: string) => {
    const el = messageRefsMap.current.get(id);
    if (!el) return;
    const scrollContainer = conversationScrollRef.current;
    if (scrollContainer) {
      const elTop = el.offsetTop - scrollContainer.offsetTop;
      scrollContainer.scrollTo({ top: elTop - 16, behavior: "smooth" });
    } else {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    if (!authInitialized || !user) return;
    try {
      const data = await apiRequest("/documents/list", {}, user);
      setDocuments(data.documents || []);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load documents.");
    }
  }, [user, authInitialized]);

  const loadSessions = useCallback(async () => {
    if (!authInitialized || !user) return;
    try {
      const data = await apiRequest("/chat/sessions", {}, user);
      setSessions(data || []);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load conversations.");
    }
  }, [user, authInitialized]);

  useEffect(() => {
    if (!firebaseAuth) {
      setAuthInitialized(true);
      return;
    }
    return onAuthStateChanged(firebaseAuth, (firebaseUser) => {
      setUser(firebaseUser);
      setAuthInitialized(true);
      
      // Show login modal every time user opens/refreshes without authentication
      if (!firebaseUser && !LOCAL_AUTH_MODE) {
        setTimeout(() => {
          setShowLoginModal(true);
        }, 1000); // Show after 1 second
      }
    });
  }, []);

  useEffect(() => { 
    if (authInitialized && user) {
      loadDocuments();
      loadSessions();
    }
  }, [authInitialized, user, loadDocuments, loadSessions]);

  const handleSignOut = () => {
    if (firebaseAuth) signOut(firebaseAuth);
  };

  const handleCopy = (messageId: string, content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopiedId(messageId);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };

  const handleSignIn = async () => {
    if (!firebaseAuth) {
      setError("Firebase sign-in is not configured.");
      return;
    }
    try {
      await signInWithPopup(firebaseAuth, new GoogleAuthProvider());
      setShowLoginModal(false);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not sign in.");
    }
  };

  const startNewConversation = () => {
    setActiveSessionId(null);
    setMessages([]);
    if (window.innerWidth <= 720) {
      setSidebarOpen(false);
      setShowBackdrop(false);
    }
  };

  const deleteSession = async (sessionId: string) => {
    setActiveMenu(null);
    setConfirmDeleteId(sessionId);
  };

  const confirmDeleteSession = async () => {
    if (!confirmDeleteId) return;
    const sessionId = confirmDeleteId;
    setConfirmDeleteId(null);
    setError("");
    try {
      await apiRequest(`/chat/sessions/${sessionId}`, { method: "DELETE" }, user);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      await loadSessions();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not delete conversation.");
    }
  };

  const startRename = (sessionId: string, currentTitle: string) => {
    setActiveMenu(null);
    setRenamingId(sessionId);
    setRenameValue(currentTitle);
    // Focus the input on next tick after render
    setTimeout(() => renameInputRef.current?.select(), 30);
  };

  const commitRename = async (sessionId: string) => {
    const trimmed = renameValue.trim();
    setRenamingId(null);
    if (!trimmed) return;
    const prev = sessions.find((s) => s.session_id === sessionId)?.title;
    if (trimmed === prev) return;
    // Optimistic update
    setSessions((current) => current.map((s) => s.session_id === sessionId ? { ...s, title: trimmed } : s));
    try {
      await apiRequest(`/chat/sessions/${sessionId}/title`, {
        method: "PUT",
        body: JSON.stringify({ title: trimmed }),
      }, user);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not rename conversation.");
      await loadSessions(); // revert on failure
    }
  };

  const exportSession = async (sessionId: string, title: string) => {
    setActiveMenu(null);
    setError("");
    try {
      const data = await apiRequest(`/chat/sessions/${sessionId}/export`, {}, user);

      // Format as human-readable plain text
      const lines: string[] = [];
      lines.push(`Conversation: ${data.title}`);
      lines.push(`Exported:     ${new Date().toLocaleString()}`);
      lines.push(`Created:      ${new Date(data.created_at).toLocaleString()}`);
      lines.push(`Updated:      ${new Date(data.updated_at).toLocaleString()}`);
      lines.push("");
      lines.push("=".repeat(60));
      lines.push("");
      for (const msg of (data.messages || [])) {
        const label = msg.type === "user" ? "You" : "vw-brain AI";
        const ts = new Date(msg.timestamp).toLocaleString();
        lines.push(`[${label}]  ${ts}`);
        lines.push(msg.content);
        if (msg.sources && msg.sources.length) {
          lines.push("");
          lines.push("Sources:");
          msg.sources.forEach((src: { filename?: string; chunk_index?: number }, i: number) => {
            lines.push(`  [${i + 1}] ${src.filename || "Document"}${src.chunk_index != null ? ` (chunk ${src.chunk_index})` : ""}`);
          });
        }
        lines.push("");
        lines.push("-".repeat(60));
        lines.push("");
      }

      const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${title.replace(/[^a-z0-9]/gi, "_").toLowerCase()}_export.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not export conversation.");
    }
  };

  const openConversation = async (sessionId: string) => {
    setBusy(true);
    setError("");
    try {
      const data = await apiRequest(`/chat/sessions/${sessionId}`, {}, user);
      setActiveSessionId(sessionId);
      setMessages((data.messages || []).map((message: { id: number; type: "user" | "assistant"; content: string; sources?: Source[] }) => ({
        id: String(message.id),
        type: message.type,
        content: message.type === "assistant" ? cleanAssistantResponse(message.content) : message.content,
        sources: message.sources,
      })));
      if (window.innerWidth <= 720) {
        setSidebarOpen(false);
        setShowBackdrop(false);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not open conversation.");
    } finally {
      setBusy(false);
    }
  };

  const uploadDocument = async (file: File) => {
    if (file.type !== "application/pdf") { setError("Only PDF files can be added to the library."); return; }

    setUploading(true);
    setUploadProgress(0);
    setUploadStep("reading");
    setUploadSuccess(null);
    setError("");

    // Simulated progress: advances through stages while the real request runs
    let currentProgress = 0;
    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        const next = prev + (prev < 30 ? 4 : prev < 60 ? 2.5 : prev < 80 ? 1.5 : prev < 90 ? 0.6 : 0);
        currentProgress = next;
        return next;
      });
      setUploadStep(currentProgress < 30 ? "reading" : currentProgress < 65 ? "uploading" : "indexing");
    }, 120);

    const formData = new FormData();
    formData.append("file", file);

    try {
      await apiRequest("/documents/upload-pdf", { method: "POST", body: formData }, user);
      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadStep("done");
      await loadDocuments();
      // Show success flash, then reset
      setUploadSuccess(file.name);
      setTimeout(() => {
        setUploadSuccess(null);
        setUploadProgress(0);
        setUploadStep(null);
        setUploading(false);
      }, 2200);
    } catch (requestError) {
      clearInterval(progressInterval);
      setError(requestError instanceof Error ? requestError.message : "Upload failed.");
      setUploadProgress(0);
      setUploadStep(null);
      setUploading(false);
    }
  };

  const handleAttachClick = () => {
    // Open library panel if closed so user can see their documents
    if (!libraryOpen) {
      setLibraryOpen(true);
      if (window.innerWidth <= 720) {
        setShowBackdrop(true);
        setSidebarOpen(false);
      }
    }
    // Trigger file input
    fileInput.current?.click();
  };

  const deleteDocument = async (documentId: string) => {
    if (!window.confirm("Delete this uploaded document?")) return;
    setError("");
    try {
      await apiRequest(`/documents/delete/${documentId}`, { method: "DELETE" }, user);
      setSelectedDocuments((current) => current.filter((id) => id !== documentId));
      await loadDocuments();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not delete document.");
    }
  };

  const submitQuery = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery || busy) return;
    const userMessage = { id: crypto.randomUUID(), type: "user" as const, content: trimmedQuery };
    setMessages((current) => [...current, userMessage]);
    setQuery(""); setBusy(true); setError("");
    try {
      let sessionId = activeSessionId;
      if (!sessionId) {
        const session = await apiRequest("/chat/sessions", {
          method: "POST",
          body: JSON.stringify({ title: trimmedQuery.slice(0, 48) }),
        }, user);
        sessionId = session.session_id;
        setActiveSessionId(sessionId);
      }
      await apiRequest(`/chat/sessions/${sessionId}/messages`, {
        method: "POST",
        body: JSON.stringify({ content: trimmedQuery, message_type: "user" }),
      }, user);
      const data = await apiRequest("/rag/search-llm", {
        method: "POST",
        body: JSON.stringify({ query: trimmedQuery, top_k: 5, document_ids: selectedDocuments.length ? selectedDocuments : null }),
      }, user);
      const assistantContent = cleanAssistantResponse(data.response);
      const assistantMessage = { id: crypto.randomUUID(), type: "assistant" as const, content: assistantContent, sources: data.sources || [] };
      setMessages((current) => [...current, assistantMessage]);
      await apiRequest(`/chat/sessions/${sessionId}/messages`, {
        method: "POST",
        body: JSON.stringify({ content: assistantContent, message_type: "assistant", sources: data.sources || [] }),
      }, user);
      await loadSessions();
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : "The assistant could not answer."); }
    finally { setBusy(false); }
  };

  const toggleDocument = (id: string) => setSelectedDocuments((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
  const filteredDocuments = documents.filter((document) => document.filename.toLowerCase().includes(search.toLowerCase()));

  const toggleSidebar = () => {
    const newState = !sidebarOpen;
    setSidebarOpen(newState);
    setShowBackdrop(newState && window.innerWidth <= 720);
    if (newState && window.innerWidth <= 720) setLibraryOpen(false);
  };

  const toggleLibrary = () => {
    const newState = !libraryOpen;
    setLibraryOpen(newState);
    setShowBackdrop(newState && window.innerWidth <= 720);
    if (newState && window.innerWidth <= 720) setSidebarOpen(false);
  };

  const closeOverlays = () => {
    if (window.innerWidth <= 720) {
      setSidebarOpen(false);
      setLibraryOpen(false);
      setShowBackdrop(false);
    }
  };

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (confirmDeleteId) { setConfirmDeleteId(null); return; }
        if (renamingId) { setRenamingId(null); return; }
        if (activeMenu) { setActiveMenu(null); return; }
        // Don't close login modal with Escape
        if (!showLoginModal) {
          closeOverlays();
        }
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [showLoginModal, activeMenu, renamingId, confirmDeleteId]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 720) {
        setShowBackdrop(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Close context menu on outside click
  useEffect(() => {
    if (!activeMenu) return;
    const close = (e: MouseEvent) => {
      const target = e.target as Element;
      if (!target.closest(".session-menu")) setActiveMenu(null);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [activeMenu]);

  return (
    <main className={`workspace-shell ${sidebarOpen ? "" : "sidebar-hidden"} ${libraryOpen ? "" : "library-hidden"}`}>
      {/* Mobile backdrop */}
      {showBackdrop && <div className="mobile-backdrop" onClick={closeOverlays} />}
      
      {/* Login Modal */}
      {showLoginModal && !LOCAL_AUTH_MODE && firebaseAuth && (
        <>
          <div className="modal-backdrop" />
          <div className="login-modal">
            <div className="modal-header">
              <div className="brand-mark"><Sparkles size={22} strokeWidth={2.5} /></div>
            </div>
            <h2>Welcome to vw-brain</h2>
            <p>Sign in with your Google account to access your private document library and start asking questions.</p>
            <button className="google-sign-in-button" onClick={handleSignIn}>
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M17.64 9.20454C17.64 8.56636 17.5827 7.95272 17.4764 7.36363H9V10.845H13.8436C13.635 11.97 13.0009 12.9231 12.0477 13.5613V15.8195H14.9564C16.6582 14.2527 17.64 11.9454 17.64 9.20454Z" fill="#4285F4"/>
                <path d="M9 18C11.43 18 13.4673 17.1941 14.9564 15.8195L12.0477 13.5613C11.2418 14.1013 10.2109 14.4204 9 14.4204C6.65591 14.4204 4.67182 12.8372 3.96409 10.71H0.957275V13.0418C2.43818 15.9831 5.48182 18 9 18Z" fill="#34A853"/>
                <path d="M3.96409 10.71C3.78409 10.17 3.68182 9.59318 3.68182 9C3.68182 8.40682 3.78409 7.82999 3.96409 7.28999V4.95818H0.957275C0.347727 6.17318 0 7.54772 0 9C0 10.4523 0.347727 11.8268 0.957275 13.0418L3.96409 10.71Z" fill="#FBBC05"/>
                <path d="M9 3.57955C10.3214 3.57955 11.5077 4.03364 12.4405 4.92545L15.0218 2.34409C13.4632 0.891818 11.4259 0 9 0C5.48182 0 2.43818 2.01682 0.957275 4.95818L3.96409 7.29C4.67182 5.16273 6.65591 3.57955 9 3.57955Z" fill="#EA4335"/>
              </svg>
              Continue with Google
            </button>
            <div className="modal-features">
              <div className="feature-item">
                <ShieldCheck size={18} />
                <span>Your documents stay private</span>
              </div>
              <div className="feature-item">
                <FileText size={18} />
                <span>Upload PDFs and ask questions</span>
              </div>
              <div className="feature-item">
                <Sparkles size={18} />
                <span>Get AI-powered answers with citations</span>
              </div>
            </div>
          </div>
        </>
      )}
      
      {/* Delete conversation confirm modal */}
      {confirmDeleteId && (
        <>
          <div className="modal-backdrop" onClick={() => setConfirmDeleteId(null)} style={{ zIndex: 210 }} />
          <div className="delete-confirm-modal">
            <div className="delete-confirm-icon"><Trash2 size={20} /></div>
            <h3>Delete conversation?</h3>
            <p>
              <strong>{sessions.find((s) => s.session_id === confirmDeleteId)?.title || "This conversation"}</strong>
              {" "}will be permanently removed and cannot be recovered.
            </p>
            <div className="delete-confirm-actions">
              <button className="delete-confirm-cancel" onClick={() => setConfirmDeleteId(null)}>Cancel</button>
              <button className="delete-confirm-ok" onClick={confirmDeleteSession}>Delete</button>
            </div>
          </div>
        </>
      )}

      {/* Mobile header - only visible on mobile */}
      <header className="mobile-header">
        <button className="mobile-header-button" aria-label="Open menu" onClick={toggleSidebar}>
          <Menu size={20} />
        </button>
        <div className="mobile-brand">
          <div className="brand-mark"><Sparkles size={15} strokeWidth={2.5} /></div>
          <span className="brand-name">vw-brain<span>.</span></span>
        </div>
        <button className="mobile-header-button" aria-label="Open library" onClick={toggleLibrary}>
          <BookOpen size={20} />
        </button>
      </header>

      {sidebarOpen && <aside className="sidebar">
        <div className="brand-row"><div className="brand-mark"><Sparkles size={17} strokeWidth={2.5} /></div><span className="brand-name">vw-brain<span>.</span></span><button className="icon-button sidebar-toggle" aria-label="Hide conversation sidebar" onClick={toggleSidebar}><X size={18} /></button></div>
        <button className="new-chat-button" onClick={startNewConversation}><Plus size={18} /> New conversation</button>
        <div className="sidebar-section-label recent-label">Recent conversations</div>
        
        {/* Conversation search */}
        <div className="convo-search">
          <Search size={13} />
          <input
            value={sessionSearch}
            onChange={(e) => setSessionSearch(e.target.value)}
            placeholder="Search conversations…"
          />
          {sessionSearch && <button className="convo-search-clear" onClick={() => setSessionSearch("")}><X size={12} /></button>}
        </div>

        <div className="recent-list">
          {(() => {
            const filtered = sessions.filter((s) =>
              s.title.toLowerCase().includes(sessionSearch.toLowerCase())
            );
            const visible = showAllSessions ? filtered : filtered.slice(0, 8);
            const formatDate = (iso: string) => {
              const d = new Date(iso);
              const now = new Date();
              const diff = now.getTime() - d.getTime();
              if (diff < 60_000) return "just now";
              if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
              if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
              if (diff < 604_800_000) return `${Math.floor(diff / 86_400_000)}d ago`;
              return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
            };

            if (!filtered.length) return (
              <p className="empty-sidebar">
                {sessionSearch ? `No conversations match "${sessionSearch}"` : "Your conversations will appear here."}
              </p>
            );

            return (
              <>
                {visible.map((session) => (
                  <div
                    className={`recent-item-wrap ${activeSessionId === session.session_id ? "active" : ""}`}
                    key={session.session_id}
                  >
                    {renamingId === session.session_id ? (
                      // Inline rename input
                      <input
                        ref={renameInputRef}
                        className="rename-input"
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onBlur={() => commitRename(session.session_id)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitRename(session.session_id);
                          if (e.key === "Escape") setRenamingId(null);
                        }}
                        maxLength={80}
                        autoFocus
                      />
                    ) : (
                      <button
                        className="recent-item-btn"
                        onClick={() => openConversation(session.session_id)}
                        title={session.title}
                      >
                        <span className="recent-item-title">{session.title}</span>
                        <span className="recent-item-meta">
                          {session.message_count > 0 && (
                            <span className="meta-count">
                              <MessageSquare size={10} />{session.message_count}
                            </span>
                          )}
                          <span className="meta-date">{formatDate(session.updated_at)}</span>
                        </span>
                      </button>
                    )}

                    {/* Context menu trigger */}
                    <div className="session-menu">
                      <button
                        className="session-menu-trigger"
                        aria-label="Conversation options"
                        onClick={(e) => {
                          e.stopPropagation();
                          setActiveMenu(activeMenu === session.session_id ? null : session.session_id);
                        }}
                      >
                        <MoreHorizontal size={15} />
                      </button>
                      {activeMenu === session.session_id && (
                        <div className="session-menu-dropdown">
                          <button onClick={() => startRename(session.session_id, session.title)}>
                            <Pencil size={13} /> Rename
                          </button>
                          <button onClick={() => exportSession(session.session_id, session.title)}>
                            <Download size={13} /> Export .txt
                          </button>
                          <button className="menu-danger" onClick={() => deleteSession(session.session_id)}>
                            <Trash2 size={13} /> Delete
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Show more / less */}
                {filtered.length > 8 && (
                  <button className="show-more-btn" onClick={() => setShowAllSessions((v) => !v)}>
                    {showAllSessions
                      ? `Show less`
                      : `Show ${filtered.length - 8} more`}
                  </button>
                )}
              </>
            );
          })()}
        </div>
        <div className="sidebar-footer"><div className="security-note"><ShieldCheck size={17} /><span><strong>Private by design</strong><small>Your documents stay in your workspace.</small></span></div><div className="account-row"><div className="avatar">{user?.email?.[0]?.toUpperCase() || "L"}</div><div className="account-copy"><strong>{user?.displayName || (LOCAL_AUTH_MODE ? "Local workspace" : "Google Account")}</strong><small>{LOCAL_AUTH_MODE ? "Local development" : user?.email || "Not signed in"}</small></div>{user && firebaseAuth ? <button className="icon-button" aria-label="Sign out" onClick={handleSignOut}><LogOut size={16} /></button> : !LOCAL_AUTH_MODE && firebaseAuth ? <button className="sign-in-button" onClick={handleSignIn}>Sign in</button> : null}</div></div>
      </aside>}
      {!sidebarOpen && <aside className="collapsed-sidebar" aria-label="Collapsed conversation sidebar"><button className="collapsed-rail-button" aria-label="Show conversation sidebar" onClick={toggleSidebar}><Menu size={19} /></button><button className="collapsed-rail-button" aria-label="New conversation" onClick={startNewConversation}><Plus size={20} /></button></aside>}

      <section className="main-panel">
        {!libraryOpen && <button className="icon-button library-restore" aria-label="Show document library" onClick={toggleLibrary}><PanelRightOpen size={19} /></button>}
        <div className={`content-grid ${libraryOpen ? "" : "library-hidden"}`}>
          <section className={`chat-column ${messages.length > 0 ? "has-messages" : ""}`}>
            <div className="chat-column-inner">
            <div className="conversation-scroll" ref={conversationScrollRef}><div className="chat-intro"><h2>Ask your library.<br /><em>See the whole picture.</em></h2></div>
            {messages.length > 0 && <div className="message-list">{messages.map((message) => <article className={`message ${message.type}`} key={message.id} ref={(el) => setMessageRef(message.id, el)}>
                      <div className="message-header">
                        <div className="message-label">{message.type === "user" ? "You" : "vw-brain AI"}</div>
                        {message.type === "assistant" && (
                          <button
                            className={`copy-btn${copiedId === message.id ? " copy-btn--done" : ""}`}
                            aria-label="Copy reply"
                            onClick={() => handleCopy(message.id, message.content)}
                          >
                            {copiedId === message.id ? <><CopyCheck size={12} />Copied</> : <><Copy size={12} />Copy</>}
                          </button>
                        )}
                      </div>
                      <div className="message-content">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            table: ({ children, ...props }) => (
                              <div className="table-scroll-wrapper">
                                <table {...props}>{children}</table>
                              </div>
                            ),
                          }}
                        >{message.content}</ReactMarkdown>
                      </div>
                      {message.sources && message.sources.length > 0 && (
                        <div className="source-row">
                          {message.sources.slice(0, 3).map((source, index) => (
                            <span className="source-pill" key={`${source.filename}-${index}`}>
                              <FileText size={13} />{source.filename || "Document"}<b>[{index + 1}]</b>
                            </span>
                          ))}
                        </div>
                      )}
                    </article>)}</div>}
            {busy && <div className="thinking"><LoaderCircle size={17} className="spin" /> Reading your library...</div>}</div>
            <ConversationRail messages={messages} onJump={jumpToMessage} />
            </div>
            <div className="composer-wrap"><div className="selection-line"><span><Check size={14} /> {selectedDocuments.length ? `${selectedDocuments.length} document${selectedDocuments.length > 1 ? "s" : ""} selected` : "Searching all documents"}</span><button onClick={() => setSelectedDocuments([])}>Clear selection</button></div><div className="composer"><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitQuery(); } }} placeholder="Ask a question about your documents..." rows={2} /><div className="composer-tools"><button className="icon-button" aria-label="Attach document" onClick={handleAttachClick}><Paperclip size={18} /></button><span className="composer-hint">Shift + Enter for a new line</span><button className="send-button" aria-label="Send question" onClick={submitQuery} disabled={!query.trim() || busy}><ArrowUp size={18} /></button></div></div></div>
          </section>
          {libraryOpen && <aside className="library-panel"><div className="library-heading"><div><span className="eyebrow">Knowledge base</span><h3>Your library <span>{documents.length}</span></h3></div><button className="icon-button library-toggle" aria-label="Hide document library" onClick={toggleLibrary}><X size={19} /></button></div>

            {/* Upload zone with progress */}
            <button
              className={`upload-zone ${uploading ? "upload-zone--busy" : ""} ${isDragOver ? "upload-zone--dragover" : ""} ${uploadStep === "done" ? "upload-zone--done" : ""}`}
              onClick={() => !uploading && fileInput.current?.click()}
              onDragOver={(e) => { e.preventDefault(); if (!uploading) setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragOver(false);
                const file = e.dataTransfer.files?.[0];
                if (file && !uploading) uploadDocument(file);
              }}
              disabled={uploading}
              aria-label="Upload PDF document"
            >
              <span className={`upload-icon ${uploading && uploadStep !== "done" ? "upload-icon--spin" : ""} ${uploadStep === "done" ? "upload-icon--done" : ""}`}>
                {uploadStep === "done" ? <Check size={21} /> : <UploadCloud size={21} />}
              </span>
              <span className="upload-zone-text">
                {!uploading && !uploadStep && (
                  <>
                    <strong>{isDragOver ? "Drop to upload" : "Add a PDF"}</strong>
                    <small>Drop it here or browse files</small>
                  </>
                )}
                {uploading && uploadStep !== "done" && (
                  <>
                    <strong>
                      {uploadStep === "reading" && "Reading file…"}
                      {uploadStep === "uploading" && "Uploading…"}
                      {uploadStep === "indexing" && "Building index…"}
                    </strong>
                    <small className="upload-step-label">
                      {uploadStep === "reading" && "Parsing PDF structure"}
                      {uploadStep === "uploading" && "Sending to server"}
                      {uploadStep === "indexing" && "Generating embeddings"}
                    </small>
                  </>
                )}
                {uploadStep === "done" && (
                  <>
                    <strong className="upload-success-text">Added to library!</strong>
                    <small className="upload-filename">{uploadSuccess}</small>
                  </>
                )}
              </span>
              {!uploading && <ChevronDown size={16} className="upload-chevron" />}
              {uploading && uploadStep !== "done" && (
                <span className="upload-pct">{Math.round(uploadProgress)}%</span>
              )}
              {/* Progress bar */}
              {uploading && (
                <span
                  className={`upload-progress-bar ${uploadStep === "done" ? "upload-progress-bar--done" : ""}`}
                  style={{ width: `${uploadProgress}%` }}
                />
              )}
            </button>

            <div className="library-search"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter documents" /></div><div className="document-list">{filteredDocuments.map((document) => <div className={`document-card ${selectedDocuments.includes(document.file_id) ? "selected" : ""}`} key={document.file_id} onClick={() => toggleDocument(document.file_id)}><div className="pdf-icon"><FileText size={19} /></div><span className="document-copy"><strong>{document.filename}</strong><small>{formatBytes(document.size_bytes)}{document.uploaded_at ? ` · ${new Date(document.uploaded_at * 1000).toLocaleDateString()}` : ""}</small></span><span className="document-check">{selectedDocuments.includes(document.file_id) ? <Check size={15} /> : <span />}</span><button className="delete-document" aria-label={`Delete ${document.filename}`} onClick={(event) => { event.stopPropagation(); deleteDocument(document.file_id); }}><Trash2 size={14} /></button></div>)}{!filteredDocuments.length && <div className="empty-library"><BookOpen size={23} /><strong>Your library is quiet.</strong><span>Add a PDF to start asking questions.</span></div>}</div><div className="library-footer"><span><span className="tiny-dot" /> All systems operational</span><span>API · 8000</span></div></aside>}
        </div>
        {/* Hidden file input - always available for both paperclip and upload zone */}
        <input ref={fileInput} type="file" accept="application/pdf" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) uploadDocument(file); event.target.value = ""; }} />
        {error && <button className="error-toast" onClick={() => setError("")}><X size={16} />{error}</button>}
      </section>
    </main>
  );
}
