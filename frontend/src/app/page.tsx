"use client";

import {
  ArrowUp, BookOpen, Check, ChevronDown, FileText,
  LoaderCircle, LogOut, Menu, MoreHorizontal,
  Paperclip, PanelRightOpen, Plus, Search, ShieldCheck,
  Sparkles, Trash2, UploadCloud, X,
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
  const [error, setError] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [authInitialized, setAuthInitialized] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [showBackdrop, setShowBackdrop] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

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

  const handleSignIn = async () => {
    if (!firebaseAuth) {
      setError("Firebase sign-in is not configured.");
      return;
    }
    try {
      await signInWithPopup(firebaseAuth, new GoogleAuthProvider());
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
    setUploading(true); setError("");
    const formData = new FormData(); formData.append("file", file);
    try { await apiRequest("/documents/upload-pdf", { method: "POST", body: formData }, user); await loadDocuments(); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : "Upload failed."); }
    finally { setUploading(false); }
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
      if (e.key === "Escape") closeOverlays();
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 720) {
        setShowBackdrop(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <main className={`workspace-shell ${sidebarOpen ? "" : "sidebar-hidden"} ${libraryOpen ? "" : "library-hidden"}`}>
      {/* Mobile backdrop */}
      {showBackdrop && <div className="mobile-backdrop" onClick={closeOverlays} />}
      
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
        <div className="recent-list">{sessions.length ? sessions.slice(0, 6).map((session) => <button className={`recent-item ${activeSessionId === session.session_id ? "active" : ""}`} key={session.session_id} onClick={() => openConversation(session.session_id)}><span>{session.title}</span><MoreHorizontal size={16} /></button>) : <p className="empty-sidebar">Your conversations will appear here.</p>}</div>
        <div className="sidebar-footer"><div className="security-note"><ShieldCheck size={17} /><span><strong>Private by design</strong><small>Your documents stay in your workspace.</small></span></div><div className="account-row"><div className="avatar">{user?.email?.[0]?.toUpperCase() || "L"}</div><div className="account-copy"><strong>{user?.displayName || (LOCAL_AUTH_MODE ? "Local workspace" : "Firebase account")}</strong><small>{LOCAL_AUTH_MODE ? "Local development" : user?.email || "Not signed in"}</small></div>{user && firebaseAuth ? <button className="icon-button" aria-label="Sign out" onClick={handleSignOut}><LogOut size={16} /></button> : !LOCAL_AUTH_MODE && firebaseAuth ? <button className="sign-in-button" onClick={handleSignIn}>Sign in</button> : null}</div></div>
      </aside>}
      {!sidebarOpen && <aside className="collapsed-sidebar" aria-label="Collapsed conversation sidebar"><button className="collapsed-rail-button" aria-label="Show conversation sidebar" onClick={toggleSidebar}><Menu size={19} /></button><button className="collapsed-rail-button" aria-label="New conversation" onClick={startNewConversation}><Plus size={20} /></button></aside>}

      <section className="main-panel">
        {!libraryOpen && <button className="icon-button library-restore" aria-label="Show document library" onClick={toggleLibrary}><PanelRightOpen size={19} /></button>}
        <div className={`content-grid ${libraryOpen ? "" : "library-hidden"}`}>
          <section className="chat-column">
            <div className="conversation-scroll"><div className="chat-intro"><h2>Ask your library.<br /><em>See the whole picture.</em></h2></div>
            {messages.length > 0 && <div className="message-list">{messages.map((message) => <article className={`message ${message.type}`} key={message.id}><div className="message-label">{message.type === "user" ? "You" : "vw-brain AI"}</div><div className="message-content"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>{message.sources && message.sources.length > 0 && <div className="source-row">{message.sources.slice(0, 3).map((source, index) => <span className="source-pill" key={`${source.filename}-${index}`}><FileText size={13} />{source.filename || "Document"}<b>[{index + 1}]</b></span>)}</div>}</article>)}</div>}
            {busy && <div className="thinking"><LoaderCircle size={17} className="spin" /> Reading your library...</div>}</div>
            <div className="composer-wrap"><div className="selection-line"><span><Check size={14} /> {selectedDocuments.length ? `${selectedDocuments.length} document${selectedDocuments.length > 1 ? "s" : ""} selected` : "Searching all documents"}</span><button onClick={() => setSelectedDocuments([])}>Clear selection</button></div><div className="composer"><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitQuery(); } }} placeholder="Ask a question about your documents..." rows={2} /><div className="composer-tools"><button className="icon-button" aria-label="Attach document" onClick={() => fileInput.current?.click()}><Paperclip size={18} /></button><span className="composer-hint">Shift + Enter for a new line</span><button className="send-button" aria-label="Send question" onClick={submitQuery} disabled={!query.trim() || busy}><ArrowUp size={18} /></button></div></div></div>
          </section>
          {libraryOpen && <aside className="library-panel"><div className="library-heading"><div><span className="eyebrow">Knowledge base</span><h3>Your library <span>{documents.length}</span></h3></div><button className="icon-button library-toggle" aria-label="Hide document library" onClick={toggleLibrary}><X size={19} /></button></div><button className="upload-zone" onClick={() => fileInput.current?.click()}><span className="upload-icon"><UploadCloud size={21} /></span><span><strong>{uploading ? "Adding document..." : "Add a PDF"}</strong><small>Drop it here or browse files</small></span><ChevronDown size={16} className="upload-chevron" /></button><input ref={fileInput} type="file" accept="application/pdf" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) uploadDocument(file); event.target.value = ""; }} /><div className="library-search"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter documents" /></div><div className="document-list">{filteredDocuments.map((document) => <div className={`document-card ${selectedDocuments.includes(document.file_id) ? "selected" : ""}`} key={document.file_id} onClick={() => toggleDocument(document.file_id)}><div className="pdf-icon"><FileText size={19} /></div><span className="document-copy"><strong>{document.filename}</strong><small>{formatBytes(document.size_bytes)}{document.uploaded_at ? ` · ${new Date(document.uploaded_at * 1000).toLocaleDateString()}` : ""}</small></span><span className="document-check">{selectedDocuments.includes(document.file_id) ? <Check size={15} /> : <span />}</span><button className="delete-document" aria-label={`Delete ${document.filename}`} onClick={(event) => { event.stopPropagation(); deleteDocument(document.file_id); }}><Trash2 size={14} /></button></div>)}{!filteredDocuments.length && <div className="empty-library"><BookOpen size={23} /><strong>Your library is quiet.</strong><span>Add a PDF to start asking questions.</span></div>}</div><div className="library-footer"><span><span className="tiny-dot" /> All systems operational</span><span>API · 8000</span></div></aside>}
        </div>
        {error && <button className="error-toast" onClick={() => setError("")}><X size={16} />{error}</button>}
      </section>
    </main>
  );
}
