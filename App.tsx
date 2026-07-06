import React, { useState, useEffect, useRef } from 'react';
import { 
  LayoutDashboard, 
  FolderOpen, 
  MessageSquare, 
  Settings, 
  UploadCloud, 
  FileText, 
  Trash2, 
  Eye, 
  BrainCircuit, 
  Database,
  Search, 
  Send,
  AlertCircle,
  CheckCircle2,
  Clock,
  Compass
} from 'lucide-react';

// API Base URL
const API_URL = 'http://127.0.0.1:8000/api';

interface Document {
  id: number;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  upload_date: string;
  status: 'processing' | 'ready' | 'failed';
  ai_summary?: string;
  ai_keywords?: string;
  category?: string;
  document_type?: string;
  embedding?: string;
}

interface Stats {
  total_documents: number;
  total_size_bytes: number;
  status_breakdown: {
    processing: number;
    ready: number;
    failed: number;
  };
  top_tags: { tag: string; count: number }[];
  type_distribution: Record<string, number>;
}

interface Message {
  sender: 'user' | 'ai';
  text: string;
  timestamp: Date;
  sources?: Document[];
  confidence_score?: number;
  page_numbers?: number[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'documents' | 'chat' | 'settings'>('dashboard');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState<Stats>({
    total_documents: 0,
    total_size_bytes: 0,
    status_breakdown: { processing: 0, ready: 0, failed: 0 },
    top_tags: [],
    type_distribution: {}
  });

  // Modal State
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  // File Upload State
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Chat State
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'ai',
      text: "Hello! I am your Vault AI. Ask me anything about your uploaded documents, and I'll search their contents and summarize the answers for you.",
      timestamp: new Date()
    }
  ]);
  const [isSendingQuery, setIsSendingQuery] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Search filter
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedChatDocIds, setSelectedChatDocIds] = useState<number[]>([]);
  const [isFilterExpanded, setIsFilterExpanded] = useState(false);

  // Fetch Data
  const fetchData = async () => {
    try {
      const docsRes = await fetch(`${API_URL}/documents`);
      if (docsRes.ok) {
        const docsData = await docsRes.json();
        setDocuments(docsData);
      }

      const statsRes = await fetch(`${API_URL}/stats`);
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }
    } catch (err) {
      console.error("Failed to connect to backend api:", err);
    }
  };

  // Initial load and periodic polling for processing files
  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    // If any document is processing, poll every 3 seconds
    const hasProcessing = documents.some(doc => doc.status === 'processing');
    if (hasProcessing) {
      const interval = setInterval(() => {
        fetchData();
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [documents]);

  // Scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Helper: Format byte sizes
  const formatBytes = (bytes: number, decimals = 2) => {
    if (!bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  // Helper: Format Dates
  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr);
    return d.toLocaleDateString(undefined, { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Drag and drop event handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const uploadFile = async (file: File) => {
    setUploadProgress('uploading');
    setErrorMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to upload document");
      }

      setUploadProgress('success');
      fetchData();
      
      // Auto close modal after 1.5 seconds on success
      setTimeout(() => {
        setIsUploadModalOpen(false);
        setUploadProgress('idle');
      }, 1500);

    } catch (err: any) {
      console.error(err);
      setUploadProgress('error');
      setErrorMessage(err.message || "An error occurred during file upload.");
    }
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent opening modal
    if (!confirm("Are you sure you want to delete this document from your vault?")) return;

    try {
      const res = await fetch(`${API_URL}/documents/${id}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        setDocuments(prev => prev.filter(d => d.id !== id));
        fetchData();
        if (selectedDoc && selectedDoc.id === id) {
          setSelectedDoc(null);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsgText = chatInput;
    setChatInput('');
    setMessages(prev => [...prev, {
      sender: 'user',
      text: userMsgText,
      timestamp: new Date()
    }]);

    setIsSendingQuery(true);

    try {
      const requestPayload: any = { query: userMsgText };
      if (selectedChatDocIds.length > 0) {
        requestPayload.document_ids = selectedChatDocIds;
      }
      if (messages.length > 0) {
        requestPayload.history = messages.map(msg => ({
          sender: msg.sender,
          text: msg.text
        }));
      }

      const res = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload)
      });

      if (!res.ok) throw new Error("AI query failed");

      const data = await res.json();
      
      setMessages(prev => [...prev, {
        sender: 'ai',
        text: data.answer,
        timestamp: new Date(),
        sources: data.source_documents,
        confidence_score: data.confidence_score,
        page_numbers: data.page_numbers
      }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, {
        sender: 'ai',
        text: "Sorry, I encountered an error communicating with the vault query engine. Please check if the backend service is running.",
        timestamp: new Date()
      }]);
    } finally {
      setIsSendingQuery(false);
    }
  };

  const startChatAboutDoc = (doc: Document) => {
    setSelectedDoc(null);
    setActiveTab('chat');
    setSelectedChatDocIds([doc.id]);
    setIsFilterExpanded(true);
    setChatInput(`Tell me about the document: ${doc.original_filename}`);
  };

  const filteredDocs = documents.filter(doc => 
    doc.original_filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (doc.category && doc.category.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (doc.document_type && doc.document_type.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (doc.ai_keywords && doc.ai_keywords.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo-container">
          <div className="logo-icon">V</div>
          <span className="logo-text">Vault AI</span>
        </div>

        <nav>
          <ul className="nav-links">
            <li>
              <button 
                className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
                onClick={() => setActiveTab('dashboard')}
              >
                <LayoutDashboard />
                Dashboard
              </button>
            </li>
            <li>
              <button 
                className={`nav-item ${activeTab === 'documents' ? 'active' : ''}`}
                onClick={() => setActiveTab('documents')}
              >
                <FolderOpen />
                My Documents
              </button>
            </li>
            <li>
              <button 
                className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
                onClick={() => setActiveTab('chat')}
              >
                <MessageSquare />
                Ask Vault AI
              </button>
            </li>
            <li>
              <button 
                className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
                onClick={() => setActiveTab('settings')}
              >
                <Settings />
                Database Settings
              </button>
            </li>
          </ul>
        </nav>

        {/* Sidebar Vault tags cloud */}
        {stats.top_tags.length > 0 && (
          <div className="sidebar-widget">
            <div className="widget-title">Top Categories</div>
            <div className="widget-tags">
              {stats.top_tags.map(t => (
                <span 
                  key={t.tag} 
                  className="tag-cloud-item"
                  onClick={() => {
                    setSearchTerm(t.tag);
                    setActiveTab('documents');
                  }}
                >
                  #{t.tag} ({t.count})
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="sidebar-footer">
          <p>Personal Knowledge Vault v1.0</p>
          <p style={{ marginTop: '4px' }}>SQLite DB Connected</p>
        </div>
      </aside>

      {/* Main Page Area */}
      <div className="main-wrapper">
        <header className="header">
          <div className="header-title">
            <h1>
              {activeTab === 'dashboard' && "Vault Dashboard"}
              {activeTab === 'documents' && "Vault Explorer"}
              {activeTab === 'chat' && "AI Search Assistant"}
              {activeTab === 'settings' && "System Configuration"}
            </h1>
          </div>
          <div className="header-actions">
            <button className="btn btn-primary" onClick={() => setIsUploadModalOpen(true)}>
              <UploadCloud size={16} />
              Upload Documents
            </button>
          </div>
        </header>

        <main className="content-body">
          {/* DASHBOARD TAB */}
          {activeTab === 'dashboard' && (
            <div>
              {/* Stats Widgets */}
              <div className="stats-grid">
                <div className="glass-card stat-card">
                  <div className="stat-icon primary">
                    <FileText />
                  </div>
                  <div className="stat-info">
                    <span className="stat-label">Total Documents</span>
                    <span className="stat-value">{stats.total_documents}</span>
                  </div>
                </div>

                <div className="glass-card stat-card">
                  <div className="stat-icon secondary">
                    <Database />
                  </div>
                  <div className="stat-info">
                    <span className="stat-label">Vault Storage</span>
                    <span className="stat-value">{formatBytes(stats.total_size_bytes)}</span>
                  </div>
                </div>

                <div className="glass-card stat-card">
                  <div className="stat-icon success">
                    <BrainCircuit />
                  </div>
                  <div className="stat-info">
                    <span className="stat-label">AI Analyzed</span>
                    <span className="stat-value">{stats.status_breakdown.ready}</span>
                  </div>
                </div>

                {stats.status_breakdown.processing > 0 && (
                  <div className="glass-card stat-card pulse-glow">
                    <div className="stat-icon warning" style={{ backgroundColor: 'rgba(245, 158, 11, 0.12)', color: 'var(--color-warning)', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                      <Clock className="spinner" style={{ borderLeftColor: 'var(--color-warning)' }} />
                    </div>
                    <div className="stat-info">
                      <span className="stat-label">Processing Queue</span>
                      <span className="stat-value">{stats.status_breakdown.processing}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Dashboard Layout Grid */}
              <div className="dashboard-grid">
                <div className="grid-main">
                  {/* Recent Documents Table */}
                  <div className="glass-card">
                    <div className="section-header">
                      <h2 className="section-title">
                        <FolderOpen size={18} />
                        Recent Uploads
                      </h2>
                      {documents.length > 5 && (
                        <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '12px' }} onClick={() => setActiveTab('documents')}>
                          View All
                        </button>
                      )}
                    </div>

                    {documents.length === 0 ? (
                      <div className="empty-state">
                        <UploadCloud className="empty-state-icon" />
                        <h3 className="empty-state-title">No documents in the vault</h3>
                        <p className="empty-state-desc">Upload text files, markdown documents, or PDFs to build your AI Personal Knowledge Vault.</p>
                        <button className="btn btn-primary" onClick={() => setIsUploadModalOpen(true)}>
                          Upload First File
                        </button>
                      </div>
                    ) : (
                      <div className="table-container">
                        <table className="document-table">
                          <thead>
                            <tr>
                              <th>Name</th>
                              <th>Size</th>
                              <th>Status</th>
                              <th>Upload Date</th>
                              <th>Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {documents.slice(0, 5).map(doc => (
                              <tr 
                                key={doc.id} 
                                style={{ cursor: 'pointer' }}
                                onClick={() => setSelectedDoc(doc)}
                              >
                                <td>
                                  <div className="doc-name-cell">
                                    <FileText className="doc-icon" size={16} />
                                    <span className="doc-name" title={doc.original_filename}>{doc.original_filename}</span>
                                  </div>
                                </td>
                                <td><span className="doc-size">{formatBytes(doc.file_size)}</span></td>
                                <td>
                                  <span className={`status-badge ${doc.status}`}>
                                    {doc.status === 'processing' && <Clock size={10} />}
                                    {doc.status === 'ready' && <CheckCircle2 size={10} />}
                                    {doc.status === 'failed' && <AlertCircle size={10} />}
                                    {doc.status}
                                  </span>
                                </td>
                                <td><span className="doc-size">{formatDate(doc.upload_date)}</span></td>
                                <td>
                                  <div style={{ display: 'flex', gap: '8px' }} onClick={e => e.stopPropagation()}>
                                    <button className="btn btn-secondary" style={{ padding: '6px' }} title="View summary" onClick={() => setSelectedDoc(doc)}>
                                      <Eye size={14} />
                                    </button>
                                    <button className="btn btn-danger-outline" style={{ padding: '6px' }} title="Delete" onClick={e => handleDelete(doc.id, e)}>
                                      <Trash2 size={14} />
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>

                {/* Sidebar Widget Panel */}
                <div className="grid-sidebar" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  <div className="glass-card">
                    <h3 className="section-title" style={{ marginBottom: '16px' }}>
                      <BrainCircuit size={18} />
                      AI Assistant Quick Start
                    </h3>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.5', marginBottom: '16px' }}>
                      Ask questions across your entire document vault. The system uses a local keyword-search index to extract the best matching sources.
                    </p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <button 
                        className="btn btn-secondary" 
                        style={{ justifyContent: 'flex-start', fontSize: '12px', textAlign: 'left' }}
                        onClick={() => {
                          setActiveTab('chat');
                          setChatInput("Give me an executive summary of all documents in the vault.");
                        }}
                      >
                        ⚡ Executive Summary
                      </button>
                      <button 
                        className="btn btn-secondary" 
                        style={{ justifyContent: 'flex-start', fontSize: '12px', textAlign: 'left' }}
                        onClick={() => {
                          setActiveTab('chat');
                          setChatInput("What are the key technical concepts described in the documents?");
                        }}
                      >
                        ⚡ Technical Concepts
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* DOCUMENTS TAB */}
          {activeTab === 'documents' && (
            <div className="glass-card">
              <div className="section-header" style={{ flexWrap: 'wrap', gap: '16px' }}>
                <h2 className="section-title">
                  <FolderOpen size={18} />
                  Vault Contents ({filteredDocs.length})
                </h2>
                
                {/* Search Bar */}
                <div style={{ display: 'flex', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '4px 12px', width: '300px' }}>
                  <Search size={16} style={{ color: 'var(--text-muted)', marginRight: '8px' }} />
                  <input 
                    type="text" 
                    placeholder="Search by name or category..." 
                    value={searchTerm}
                    onChange={e => setSearchTerm(e.target.value)}
                    style={{ background: 'transparent', border: 'none', color: 'white', width: '100%', outline: 'none', fontSize: '13px', padding: '6px 0' }}
                  />
                  {searchTerm && (
                    <button style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '12px' }} onClick={() => setSearchTerm('')}>
                      ✕
                    </button>
                  )}
                </div>
              </div>

              {filteredDocs.length === 0 ? (
                <div className="empty-state">
                  <FolderOpen className="empty-state-icon" />
                  <h3 className="empty-state-title">{searchTerm ? "No matching files found" : "Your vault is empty"}</h3>
                  <p className="empty-state-desc">{searchTerm ? "Try searching for a different keyword or extension." : "Drag and drop files to get started."}</p>
                </div>
              ) : (
                <div className="table-container">
                  <table className="document-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Size</th>
                        <th>Category</th>
                        <th>Document Type</th>
                        <th>Status</th>
                        <th>Upload Date</th>
                        <th>AI Tags</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredDocs.map(doc => (
                        <tr 
                          key={doc.id} 
                          style={{ cursor: 'pointer' }}
                          onClick={() => setSelectedDoc(doc)}
                        >
                          <td>
                            <div className="doc-name-cell">
                              <FileText className="doc-icon" size={16} />
                              <span className="doc-name" title={doc.original_filename}>{doc.original_filename}</span>
                            </div>
                          </td>
                          <td><span className="doc-size">{formatBytes(doc.file_size)}</span></td>
                          <td>
                            {doc.category ? (
                              <span className="category-badge">{doc.category}</span>
                            ) : (
                              <span className="doc-size" style={{ fontStyle: 'italic', fontSize: '12px' }}>
                                {doc.status === 'processing' ? 'Processing...' : 'N/A'}
                              </span>
                            )}
                          </td>
                          <td>
                            {doc.document_type ? (
                              <span className="document-type-badge">{doc.document_type}</span>
                            ) : (
                              <span className="doc-size" style={{ fontStyle: 'italic', fontSize: '12px' }}>
                                {doc.status === 'processing' ? 'Processing...' : 'N/A'}
                              </span>
                            )}
                          </td>
                          <td>
                            <span className={`status-badge ${doc.status}`}>
                              {doc.status === 'processing' && <Clock size={10} />}
                              {doc.status === 'ready' && <CheckCircle2 size={10} />}
                              {doc.status === 'failed' && <AlertCircle size={10} />}
                              {doc.status}
                            </span>
                          </td>
                          <td><span className="doc-size">{formatDate(doc.upload_date)}</span></td>
                          <td>
                            <div>
                              {doc.ai_keywords ? (
                                doc.ai_keywords.split(',').slice(0, 3).map(tag => (
                                  <span key={tag} className="tag-badge">#{tag.trim()}</span>
                                ))
                              ) : (
                                <span className="doc-size" style={{ fontStyle: 'italic', fontSize: '12px' }}>None</span>
                              )}
                            </div>
                          </td>
                          <td>
                            <div style={{ display: 'flex', gap: '8px' }} onClick={e => e.stopPropagation()}>
                              <button className="btn btn-secondary" style={{ padding: '6px' }} title="View details" onClick={() => setSelectedDoc(doc)}>
                                <Eye size={14} />
                              </button>
                              <button className="btn btn-danger-outline" style={{ padding: '6px' }} title="Delete file" onClick={e => handleDelete(doc.id, e)}>
                                <Trash2 size={14} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
          {/* CHAT TAB */}
          {activeTab === 'chat' && (
            <div className="glass-card ask-container" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 150px)' }}>
              {/* Document Filter Selector */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '12px 20px', borderBottom: '1px solid rgba(255,255,255,0.06)', backgroundColor: 'rgba(0,0,0,0.15)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                  <button
                    type="button"
                    onClick={() => setIsFilterExpanded(prev => !prev)}
                    className="btn btn-secondary"
                    style={{ fontSize: '12px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}
                  >
                    🔍 Select Documents (Optional)
                    {selectedChatDocIds.length > 0 && (
                      <span style={{ backgroundColor: 'var(--accent-secondary)', color: 'black', borderRadius: '10px', padding: '2px 6px', fontSize: '10px', fontWeight: 'bold' }}>
                        {selectedChatDocIds.length}
                      </span>
                    )}
                    <span style={{ fontSize: '9px', marginLeft: '4px' }}>{isFilterExpanded ? '▲' : '▼'}</span>
                  </button>

                  {selectedChatDocIds.length > 0 && (
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                        {selectedChatDocIds.length} active
                      </span>
                      <button
                        type="button"
                        onClick={() => setSelectedChatDocIds([])}
                        className="tag-badge"
                        style={{
                          cursor: 'pointer',
                          border: '1px solid rgba(239, 68, 68, 0.3)',
                          backgroundColor: 'rgba(239, 68, 68, 0.05)',
                          color: '#fca5a5',
                          padding: '4px 8px',
                          borderRadius: '6px'
                        }}
                      >
                        Clear
                      </button>
                    </div>
                  )}
                </div>

                {isFilterExpanded && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
                    {documents.filter(d => d.status === 'ready').map(doc => {
                      const isSelected = selectedChatDocIds.includes(doc.id);
                      return (
                        <button
                          key={doc.id}
                          type="button"
                          onClick={() => {
                            setSelectedChatDocIds(prev =>
                              isSelected ? prev.filter(id => id !== doc.id) : [...prev, doc.id]
                            );
                          }}
                          className="tag-badge"
                          style={{
                            cursor: 'pointer',
                            border: isSelected ? '1px solid var(--accent-secondary)' : '1px solid rgba(255,255,255,0.1)',
                            backgroundColor: isSelected ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255,255,255,0.02)',
                            color: isSelected ? 'var(--accent-secondary)' : 'var(--text-secondary)',
                            transition: 'all 0.2s',
                            padding: '4px 8px',
                            borderRadius: '6px'
                          }}
                        >
                          {isSelected ? '✓ ' : '+ '} {doc.original_filename}
                        </button>
                      );
                    })}
                    {documents.filter(d => d.status === 'ready').length === 0 && (
                      <span style={{ fontSize: '12px', fontStyle: 'italic', color: 'var(--text-muted)' }}>No ready documents in vault.</span>
                    )}
                  </div>
                )}
              </div>

              <div className="chat-history" style={{ flex: 1, overflowY: 'auto' }}>
                {messages.map((msg, i) => (
                  <div key={i} className={`chat-message ${msg.sender}`}>
                    <div className="message-avatar">
                      {msg.sender === 'user' ? 'U' : 'AI'}
                    </div>
                    <div className="message-content">
                      <p>{msg.text}</p>
                      
                      {msg.sender === 'ai' && msg.confidence_score !== undefined && msg.confidence_score > 0 && (
                        <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--accent-secondary)', fontWeight: 500 }}>
                          Match Confidence: {(msg.confidence_score * 100).toFixed(0)}%
                        </div>
                      )}
                      
                      {msg.sources && msg.sources.length > 0 && (
                        <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                          <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Retrieved Sources:
                          </span>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '6px' }}>
                            {msg.sources.map(src => (
                              <button 
                                key={src.id} 
                                className="tag-badge" 
                                style={{ cursor: 'pointer', border: '1px solid rgba(6, 182, 212, 0.2)', backgroundColor: 'rgba(6, 182, 212, 0.05)', color: 'var(--accent-secondary)' }}
                                onClick={() => setSelectedDoc(src)}
                              >
                                📄 {src.original_filename}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {isSendingQuery && (
                  <div className="chat-message ai">
                    <div className="message-avatar">AI</div>
                    <div className="message-content" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }} />
                      <span style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>Searching vault and synthesizing response...</span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
 
              {/* Chat Input Form */}
              <form onSubmit={handleQuerySubmit} className="chat-input-container">
                <input 
                  type="text" 
                  className="chat-input"
                  placeholder="Ask a question about your vault documents..."
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  disabled={isSendingQuery}
                />
                <button type="submit" className="btn btn-primary" style={{ padding: '10px 16px' }} disabled={isSendingQuery || !chatInput.trim()}>
                  <Send size={16} />
                </button>
              </form>
            </div>
          )}

          {/* SETTINGS TAB */}
          {activeTab === 'settings' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div className="glass-card">
                <h2 className="section-title" style={{ marginBottom: '16px' }}>
                  <Database size={18} />
                  Vault Database Statistics
                </h2>
                <div className="modal-meta-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', marginBottom: '24px' }}>
                  <div className="modal-meta-item">
                    <span className="modal-section-title">Database Engine</span>
                    <div className="modal-section-text">SQLite 3 (Local File)</div>
                  </div>
                  <div className="modal-meta-item">
                    <span className="modal-section-title">Database Path</span>
                    <div className="modal-section-text">backend/vault.db</div>
                  </div>
                  <div className="modal-meta-item">
                    <span className="modal-section-title">Uploads Directory</span>
                    <div className="modal-section-text">backend/uploads/</div>
                  </div>
                  <div className="modal-meta-item">
                    <span className="modal-section-title">API Connection Status</span>
                    <div className="modal-section-text" style={{ color: 'var(--color-success)' }}>● Connected to FastAPI (Port 8000)</div>
                  </div>
                </div>
              </div>

              <div className="glass-card">
                <h2 className="section-title" style={{ marginBottom: '16px', color: 'var(--color-danger)' }}>
                  <AlertCircle size={18} />
                  Danger Zone
                </h2>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '20px', lineHeight: '1.6' }}>
                  To clear all your documents, summaries, and index points, you can delete files individually from the dashboard. Currently, backend/vault.db stores all schema indices. Deleting a document will automatically clear its record from the database and remove the raw file from disk.
                </p>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <button 
                    className="btn btn-secondary" 
                    onClick={() => {
                      if (confirm("Reset layout states?")) {
                        setSearchTerm('');
                        setMessages([{
                          sender: 'ai',
                          text: "Hello! I am your Vault AI. Ask me anything about your uploaded documents, and I'll search their contents and summarize the answers for you.",
                          timestamp: new Date()
                        }]);
                        alert("Settings reset completed.");
                      }
                    }}
                  >
                    Reset Interface State
                  </button>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* DOCUMENT DETAILS MODAL */}
      {selectedDoc && (
        <div className="modal-overlay" onClick={() => setSelectedDoc(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedDoc(null)}>✕</button>
            <h2 className="modal-title">{selectedDoc.original_filename}</h2>

            <div className="modal-meta-grid" style={{ marginBottom: '24px' }}>
              <div className="modal-meta-item">
                <span className="modal-section-title">File Size</span>
                <div className="modal-section-text">{formatBytes(selectedDoc.file_size)}</div>
              </div>
              <div className="modal-meta-item">
                <span className="modal-section-title">Upload Date</span>
                <div className="modal-section-text">{formatDate(selectedDoc.upload_date)}</div>
              </div>
              <div className="modal-meta-item">
                <span className="modal-section-title">Category</span>
                <div className="modal-section-text" style={{ marginTop: '4px' }}>
                  {selectedDoc.category ? (
                    <span className="category-badge">{selectedDoc.category}</span>
                  ) : (
                    <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                      {selectedDoc.status === 'processing' ? 'Processing...' : 'N/A'}
                    </span>
                  )}
                </div>
              </div>
              <div className="modal-meta-item">
                <span className="modal-section-title">Document Type</span>
                <div className="modal-section-text" style={{ marginTop: '4px' }}>
                  {selectedDoc.document_type ? (
                    <span className="document-type-badge">{selectedDoc.document_type}</span>
                  ) : (
                    <span style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                      {selectedDoc.status === 'processing' ? 'Processing...' : 'N/A'}
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="modal-section">
              <span className="modal-section-title">Processing Status</span>
              <div style={{ marginTop: '8px' }}>
                <span className={`status-badge ${selectedDoc.status}`}>
                  {selectedDoc.status === 'processing' && <Clock size={10} />}
                  {selectedDoc.status === 'ready' && <CheckCircle2 size={10} />}
                  {selectedDoc.status === 'failed' && <AlertCircle size={10} />}
                  {selectedDoc.status}
                </span>
              </div>
            </div>

            {selectedDoc.ai_keywords && (
              <div className="modal-section">
                <span className="modal-section-title">AI Extracted Tags</span>
                <div style={{ marginTop: '8px' }}>
                  {selectedDoc.ai_keywords.split(',').map(tag => (
                    <span key={tag} className="tag-badge" style={{ fontSize: '12px', padding: '4px 10px' }}>
                      #{tag.trim()}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="modal-section">
              <span className="modal-section-title">AI Generated Summary</span>
              <div className="modal-section-text" style={{ fontSize: '14px', lineHeight: '1.6' }}>
                {selectedDoc.ai_summary || (
                  selectedDoc.status === 'processing' 
                    ? "Vault AI is currently reading the document contents and generating a summary. Please wait..."
                    : "No summary available for this file."
                )}
              </div>
            </div>

            {/* Semantic Vector Embedding Viewer */}
            {selectedDoc.status === 'ready' && selectedDoc.embedding && (
              <div className="modal-section">
                <span className="modal-section-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Compass size={14} className="doc-icon" />
                  Semantic Embedding Vector Placeholder
                </span>
                <div className="modal-section-text" style={{ fontSize: '12px', fontFamily: 'monospace', maxHeight: '90px', overflowY: 'auto', backgroundColor: '#060a12', color: '#38bdf8', padding: '12px', border: '1px solid rgba(56, 189, 248, 0.15)', whiteSpace: 'pre-wrap' }}>
                  {(() => {
                    try {
                      const vector = JSON.parse(selectedDoc.embedding);
                      if (Array.isArray(vector)) {
                        return `Dimensions: ${vector.length} coordinates\nVector Array Preview:\n[ ${vector.slice(0, 8).join(', ')}, ... ]`;
                      }
                      return "Embedding structure matches string format but is not an array.";
                    } catch (e) {
                      return `Raw placeholder: ${selectedDoc.embedding}`;
                    }
                  })()}
                </div>
              </div>
            )}

            {selectedDoc.status === 'ready' && (
              <div style={{ marginTop: '24px', display: 'flex', gap: '12px' }}>
                <button 
                  className="btn btn-primary" 
                  style={{ flex: 1 }}
                  onClick={() => startChatAboutDoc(selectedDoc)}
                >
                  <MessageSquare size={16} />
                  Ask AI About This File
                </button>
                <button 
                  className="btn btn-secondary"
                  onClick={() => setSelectedDoc(null)}
                >
                  Close
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* UPLOAD MODAL */}
      {isUploadModalOpen && (
        <div className="modal-overlay" onClick={() => setIsUploadModalOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setIsUploadModalOpen(false)}>✕</button>
            <h2 className="modal-title">Upload Documents</h2>
            
            <div 
              className={`dropzone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={triggerFileInput}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                className="dropzone-file-input" 
                onChange={handleFileChange}
                accept=".txt,.md,.pdf,.json,.csv"
              />
              <div className="dropzone-icon">
                <UploadCloud size={28} />
              </div>
              <div className="dropzone-text">
                <span className="dropzone-title">Drag & drop files here or click to browse</span>
                <span className="dropzone-desc">Supports PDF, TXT, Markdown, JSON, and CSV (max 10MB)</span>
              </div>
            </div>

            {uploadProgress === 'uploading' && (
              <div style={{ marginTop: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
                <div className="spinner" />
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Uploading document...</span>
              </div>
            )}

            {uploadProgress === 'success' && (
              <div style={{ marginTop: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: 'var(--color-success)' }}>
                <CheckCircle2 size={20} />
                <span style={{ fontSize: '14px', fontWeight: 500 }}>Upload successful! Processing file...</span>
              </div>
            )}

            {uploadProgress === 'error' && (
              <div style={{ marginTop: '24px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                <AlertCircle size={20} style={{ color: 'var(--color-danger)', flexShrink: 0 }} />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '14px', fontWeight: 600, color: '#fca5a5' }}>Upload Failed</span>
                  <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{errorMessage}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
