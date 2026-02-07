import { useState, useEffect, useCallback, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { useDropzone, type FileRejection } from "react-dropzone";
import { toast } from "sonner";
import { PageShell } from "@/components/layout/PageShell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { MessageSquare, FileText, Brain, Plus, Upload, Search, Trash2, ChevronDown, ChevronRight, ScanLine, AlertCircle, CheckCircle, Clock, Loader2, Server, Cpu, MessageCircle } from "lucide-react";
import { getSettings, updateSettings, chatStream, listDocuments, uploadDocument, scanDocuments, getDocumentStats, searchMemories, addMemory, deleteMemory, getHealth, testConnections, getChatHistory, type ChatMessage, type Source, type Document, type DocumentStatsResponse, type MemoryResult, type ChatHistoryItem, type ConnectionTestResult } from "@/lib/api";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatStore, type Message } from "@/stores/useChatStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { Skeleton } from "@/components/ui/skeleton";

type PageId = "chat" | "documents" | "memory" | "settings";

const MAX_INPUT_LENGTH = 2000;

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown
        components={{
          code({ className, children, ...props }) {
            const isInline = !className;
            return isInline ? (
              <code className="bg-muted px-1 py-0.5 rounded text-sm font-mono" {...props}>
                {children}
              </code>
            ) : (
              <pre className="bg-muted p-3 rounded-lg overflow-x-auto my-2">
                <code className="text-sm font-mono" {...props}>
                  {children}
                </code>
              </pre>
            );
          },
          ul({ children }) {
            return <ul className="list-disc pl-5 my-2">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="list-decimal pl-5 my-2">{children}</ol>;
          },
          li({ children }) {
            return <li className="my-0.5">{children}</li>;
          },
          p({ children }) {
            return <p className="my-2">{children}</p>;
          },
          h1({ children }) {
            return <h1 className="text-xl font-bold my-3">{children}</h1>;
          },
          h2({ children }) {
            return <h2 className="text-lg font-bold my-2">{children}</h2>;
          },
          h3({ children }) {
            return <h3 className="text-base font-bold my-2">{children}</h3>;
          },
          blockquote({ children }) {
            return <blockquote className="border-l-2 border-muted-foreground pl-3 italic my-2">{children}</blockquote>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function MessageContent({ message }: { message: Message }) {
  if (message.role === "assistant") {
    return <MarkdownContent content={message.content} />;
  }
  return <span className="whitespace-pre-wrap">{message.content}</span>;
}

function ChatPage() {
  const {
    messages,
    input,
    isStreaming,
    inputError,
    expandedSources,
    setInput,
    setIsStreaming,
    setAbortFn,
    setInputError,
    addMessage,
    updateMessage,
    toggleSource,
    stopStreaming,
  } = useChatStore();

  // Chat history state
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(true);
  const [chatHistoryError, setChatHistoryError] = useState<string | null>(null);

  useEffect(() => {
    // Load chat history from localStorage
    setIsChatLoading(true);
    setChatHistoryError(null);
    try {
      const history = getChatHistory();
      setChatHistory(history);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load chat history";
      setChatHistoryError(errorMessage);
      console.error("Failed to load chat history:", err);
    } finally {
      setIsChatLoading(false);
    }
  }, []);

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    if (input.length > MAX_INPUT_LENGTH) {
      setInputError(`Input exceeds maximum length of ${MAX_INPUT_LENGTH} characters`);
      return;
    }

    const userMessage = {
      id: Date.now().toString(),
      role: "user" as const,
      content: input.trim(),
    };

    setIsStreaming(true);

    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage = {
      id: assistantMessageId,
      role: "assistant" as const,
      content: "",
    };

    const chatMessages: ChatMessage[] = [
      ...messages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content: userMessage.content },
    ];

    const abort = chatStream(chatMessages, {
      onMessage: (chunk) => {
        const currentMessages = useChatStore.getState().messages;
        const currentMsg = currentMessages.find(m => m.id === assistantMessageId);
        updateMessage(assistantMessageId, { content: (currentMsg?.content || "") + chunk });
      },
      onSources: (sources) => {
        updateMessage(assistantMessageId, { sources });
      },
      onError: (error) => {
        console.error("Chat stream error:", error);
        updateMessage(assistantMessageId, { error: error.message });
        setIsStreaming(false);
        setAbortFn(null);
      },
      onComplete: () => {
        setIsStreaming(false);
        setAbortFn(null);
      },
    });

    setAbortFn(abort);
    addMessage(userMessage);
    addMessage(assistantMessage);

    setInput("");
    setInputError(null);
  };

  const handleStop = () => {
    stopStreaming();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setInput(value);
    if (value.length > MAX_INPUT_LENGTH) {
      setInputError(`Input exceeds maximum length of ${MAX_INPUT_LENGTH} characters`);
    } else {
      setInputError(null);
    }
  };

  const handleToggleSource = (sourceId: string) => {
    toggleSource(sourceId);
  };

  const latestAssistantMessage = messages
    .filter((m) => m.role === "assistant")
    .pop();
  const hasSources = latestAssistantMessage?.sources && latestAssistantMessage.sources.length > 0;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Chat</h1>
          <p className="text-muted-foreground mt-1">Ask questions and get AI-powered answers</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => useChatStore.getState().clearMessages()}>
          <Plus className="w-4 h-4 mr-2" />
          New Chat
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Tabs defaultValue="active" className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="active">Active Chats</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="active" className="space-y-4">
          {messages.length > 0 && (
            <Card className="min-h-[300px] max-h-[500px] overflow-y-auto">
              <CardContent className="space-y-4 pt-6">
                {messages.map((message, index) => (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-4 ${
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted"
                      }`}
                    >
                      <div className="text-sm">
                        <MessageContent message={message} />
                        {isStreaming && index === messages.length - 1 && message.role === "assistant" && (
                          <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse"></span>
                        )}
                      </div>
                      {message.error && (
                        <div className="mt-2 text-xs text-destructive">
                          Error: {message.error}
                        </div>
                      )}
                      {message.stopped && (
                        <div className="mt-2 text-xs text-muted-foreground italic">
                          [stopped]
                        </div>
                      )}
                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-border/50">
                          <p className="text-xs font-medium mb-1">Sources:</p>
                          <div className="flex flex-wrap gap-1">
                            {message.sources.map((source) => (
                              <Badge key={source.id} variant="secondary" className="text-xs">
                                {source.filename}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                {messages.length === 0 ? "Start a New Conversation" : "Continue Chatting"}
              </CardTitle>
              <CardDescription>
                Type your question below to start chatting with the AI
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Textarea
                  placeholder="Ask anything about your documents..."
                  className={`min-h-[120px] resize-none ${inputError ? "border-destructive focus-visible:ring-destructive" : ""}`}
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  disabled={isStreaming}
                  maxLength={MAX_INPUT_LENGTH}
                />
                <div className="flex justify-between items-center">
                  {inputError ? (
                    <span className="text-xs text-destructive">{inputError}</span>
                  ) : (
                    <span className="text-xs text-muted-foreground"></span>
                  )}
                  <span className={`text-xs ${input.length > MAX_INPUT_LENGTH ? "text-destructive" : "text-muted-foreground"}`}>
                    {input.length}/{MAX_INPUT_LENGTH}
                  </span>
                </div>
              </div>
              <div className="flex justify-end">
                {isStreaming ? (
                  <Button variant="destructive" onClick={handleStop}>
                    Stop
                  </Button>
                ) : (
                  <Button onClick={handleSend} disabled={!input.trim()}>
                    Send Message
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            {["Previous Chat 1", "Previous Chat 2"].map((chat, i) => (
              <Card key={i} className="cursor-pointer hover:border-primary/50 transition-colors">
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-muted-foreground" />
                    <CardTitle className="text-sm font-medium">{chat}</CardTitle>
                  </div>
                  <CardDescription className="text-xs">
                    Last active 2 hours ago
                  </CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Chat History</CardTitle>
              <CardDescription>View your past conversations</CardDescription>
            </CardHeader>
            <CardContent>
              {isChatLoading ? (
                <div className="space-y-4">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="flex items-center gap-4">
                      <Skeleton className="h-10 w-10 rounded-full" />
                      <div className="space-y-2 flex-1">
                        <Skeleton className="h-4 w-[200px]" />
                        <Skeleton className="h-3 w-[150px]" />
                      </div>
                      <Skeleton className="h-3 w-[80px]" />
                    </div>
                  ))}
                </div>
              ) : chatHistoryError ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <AlertCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
                  <p className="text-muted-foreground">Failed to load chat history.</p>
                  <p className="text-xs text-muted-foreground/70 mt-1">
                    {chatHistoryError}
                  </p>
                </div>
              ) : chatHistory.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <MessageSquare className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">No chat history yet.</p>
                  <p className="text-xs text-muted-foreground/70 mt-1">
                    Start a conversation to see it here.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {chatHistory.map((chat) => (
                    <div key={chat.id} className="flex items-center gap-4 p-3 rounded-lg hover:bg-muted/50 cursor-pointer transition-colors">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <MessageSquare className="h-5 w-5 text-primary" />
                      </div>
                      <div className="flex-1">
                        <p className="font-medium">{chat.title}</p>
                        <p className="text-sm text-muted-foreground">Last active {chat.lastActive}</p>
                      </div>
                      <span className="text-xs text-muted-foreground">{chat.messageCount} messages</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
        </div>

        <div className="lg:col-span-1">
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="text-lg">Sources</CardTitle>
              <CardDescription>
                {hasSources
                  ? `${latestAssistantMessage!.sources!.length} source(s) for the latest response`
                  : "Sources will appear here after the AI responds"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {hasSources ? (
                <ScrollArea className="h-[400px] pr-4">
                  <div className="space-y-3">
                      {latestAssistantMessage!.sources!.map((source: Source) => {
                      const isExpanded = expandedSources.has(source.id);
                      return (
                        <Card key={source.id} className="border-border/50">
                          <div
                            className="p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                            onClick={() => handleToggleSource(source.id)}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <FileText className="w-4 h-4 text-muted-foreground" />
                                <span className="text-sm font-medium truncate max-w-[180px]">
                                  {source.filename}
                                </span>
                              </div>
                              <div className="flex items-center gap-2">
                                {source.score && (
                                  <Badge variant="secondary" className="text-xs">
                                    {(source.score * 100).toFixed(0)}%
                                  </Badge>
                                )}
                                {isExpanded ? (
                                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                                ) : (
                                  <ChevronRight className="w-4 h-4 text-muted-foreground" />
                                )}
                              </div>
                            </div>
                          </div>
                          {isExpanded && (
                            <div className="px-3 pb-3">
                              <div className="pt-2 border-t border-border/50">
                                <p className="text-xs text-muted-foreground whitespace-pre-wrap">
                                  {source.snippet || "No content available"}
                                </p>
                              </div>
                            </div>
                          )}
                        </Card>
                      );
                    })}
                  </div>
                </ScrollArea>
              ) : (
                <div className="flex flex-col items-center justify-center h-[200px] text-center">
                  <FileText className="w-8 h-8 text-muted-foreground/50 mb-2" />
                  <p className="text-sm text-muted-foreground">No sources available</p>
                  <p className="text-xs text-muted-foreground/70 mt-1">
                    Ask a question to see relevant sources
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState<DocumentStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [isUploading, setIsUploading] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [totalFiles, setTotalFiles] = useState(0);
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([]);

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await listDocuments();
      setDocuments(response.documents);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const response = await getDocumentStats();
      setStats(response);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  }, []);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([fetchDocuments(), fetchStats()]);
      setLoading(false);
    };
    loadData();
  }, [fetchDocuments, fetchStats]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setIsUploading(true);
    setRejectedFiles([]);
    const newProgress: Record<string, number> = {};

    for (const file of acceptedFiles) {
      newProgress[file.name] = 0;
    }
    setUploadProgress(newProgress);
    setTotalFiles(acceptedFiles.length);
    setCurrentFileIndex(0);

    try {
      for (let i = 0; i < acceptedFiles.length; i++) {
        const file = acceptedFiles[i];
        setCurrentFileIndex(i + 1);
        try {
          await uploadDocument(file, (progress) => {
            setUploadProgress((prev) => ({
              ...prev,
              [file.name]: progress,
            }));
          });
        } catch (err) {
          // Reset progress for failed file
          setUploadProgress((prev) => ({
            ...prev,
            [file.name]: 0,
          }));
          throw err;
        }
      }
      toast.success(`Uploaded ${acceptedFiles.length} file(s) successfully`);
      try {
        await Promise.all([fetchDocuments(), fetchStats()]);
      } catch (err) {
        console.error("Failed to refresh documents/stats after upload:", err);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
      setUploadProgress({});
      setCurrentFileIndex(0);
      setTotalFiles(0);
    }
  }, [fetchDocuments, fetchStats]);

  const onDropRejected = useCallback((rejected: FileRejection[]) => {
    const rejectedNames = rejected.map((r) => `${r.file.name} (${r.errors.map((e) => e.message).join(', ')})`);
    setRejectedFiles(rejectedNames);
    rejectedNames.forEach((name) => {
      toast.error(`File rejected: ${name}`);
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/markdown': ['.md'],
    },
    maxSize: MAX_FILE_SIZE,
    disabled: isUploading,
  });

  const handleScan = async () => {
    setIsScanning(true);
    try {
      const result = await scanDocuments();
      toast.success(`Scan complete: ${result.added} new document(s) added, ${result.scanned} scanned`);
      await Promise.all([fetchDocuments(), fetchStats()]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setIsScanning(false);
    }
  };

  // Handle search with debounce
  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value);
    setIsSearching(true);

    // Clear existing timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Set new timeout to stop searching after debounce
    searchTimeoutRef.current = setTimeout(() => {
      setIsSearching(false);
    }, 300);
  }, []);

  useEffect(() => {
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, []);

  const filteredDocuments = documents.filter((doc) =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "processed":
        return (
          <Badge variant="default" className="bg-green-500">
            <CheckCircle className="w-3 h-3 mr-1" />
            Processed
          </Badge>
        );
      case "processing":
        return (
          <Badge variant="secondary">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Processing
          </Badge>
        );
      case "pending":
        return (
          <Badge variant="outline">
            <Clock className="w-3 h-3 mr-1" />
            Pending
          </Badge>
        );
      case "error":
        return (
          <Badge variant="destructive">
            <AlertCircle className="w-3 h-3 mr-1" />
            Error
          </Badge>
        );
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "Unknown";
    return new Date(dateStr).toLocaleDateString();
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Documents</h1>
          <p className="text-muted-foreground mt-1">Manage your knowledge base documents</p>
        </div>
        <Button onClick={handleScan} disabled={isScanning}>
          {isScanning ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <ScanLine className="w-4 h-4 mr-2" />
          )}
          Scan Directory
        </Button>
      </div>

      {stats && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Documents</CardDescription>
              <CardTitle className="text-3xl">{stats.total_documents}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Chunks</CardDescription>
              <CardTitle className="text-3xl">{stats.total_chunks}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Size</CardDescription>
              <CardTitle className="text-3xl">{formatFileSize(stats.total_size_bytes)}</CardTitle>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Processed</CardDescription>
              <CardTitle className="text-3xl">{stats.documents_by_status?.processed || 0}</CardTitle>
            </CardHeader>
          </Card>
        </div>
      )}

      <Card
        {...getRootProps()}
        className={`border-2 border-dashed cursor-pointer transition-colors ${
          isDragActive ? "border-primary bg-primary/5" : "border-border"
        } ${isUploading ? "opacity-50 pointer-events-none" : ""}`}
      >
        <input {...getInputProps()} />
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center text-center">
            <Upload className="w-12 h-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium">
              {isDragActive ? "Drop files here..." : "Drag & drop files here, or click to select"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Supports PDF, DOCX, TXT, MD files (max 50MB each)
            </p>
          </div>
        </CardContent>
      </Card>

      {Object.keys(uploadProgress).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Upload Progress</CardTitle>
            {totalFiles > 1 && (
              <CardDescription>
                File {currentFileIndex} of {totalFiles}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.entries(uploadProgress).map(([filename, progress]) => (
              <div key={filename} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="truncate max-w-[300px]">{filename}</span>
                  <span>{progress > 0 ? `${progress}%` : 'Uploading...'}</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {rejectedFiles.length > 0 && (
        <div className="p-4 bg-amber-500/10 text-amber-700 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Some files were rejected:</span>
          </div>
          <ul className="list-disc pl-5 space-y-1">
            {rejectedFiles.map((file, index) => (
              <li key={index} className="text-sm">{file}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
          />
          {isSearching && (
            <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground animate-spin" />
          )}
        </div>
        <Badge variant="secondary">{filteredDocuments.length} documents</Badge>
      </div>

      {loading ? (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-4 font-medium">Filename</th>
                    <th className="text-left p-4 font-medium">Status</th>
                    <th className="text-left p-4 font-medium">Chunks</th>
                    <th className="text-left p-4 font-medium">Size</th>
                    <th className="text-left p-4 font-medium">Uploaded</th>
                    <th className="text-right p-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {[...Array(5)].map((_, i) => (
                    <tr key={i} className="border-b">
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <Skeleton className="h-4 w-4" />
                          <Skeleton className="h-4 w-[180px]" />
                        </div>
                      </td>
                      <td className="p-4"><Skeleton className="h-5 w-[80px]" /></td>
                      <td className="p-4"><Skeleton className="h-4 w-[40px]" /></td>
                      <td className="p-4"><Skeleton className="h-4 w-[60px]" /></td>
                      <td className="p-4"><Skeleton className="h-4 w-[80px]" /></td>
                      <td className="p-4 text-right"><Skeleton className="h-8 w-8 ml-auto" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      ) : filteredDocuments.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">
              {searchQuery ? "No documents match your search" : "No documents yet. Upload some files to get started."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-4 font-medium">Filename</th>
                    <th className="text-left p-4 font-medium">Status</th>
                    <th className="text-left p-4 font-medium">Chunks</th>
                    <th className="text-left p-4 font-medium">Size</th>
                    <th className="text-left p-4 font-medium">Uploaded</th>
                    <th className="text-right p-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredDocuments.map((doc) => (
                    <tr key={doc.id} className="border-b hover:bg-muted/50">
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-muted-foreground" />
                          <span className="font-medium truncate max-w-[200px]">{doc.filename}</span>
                        </div>
                      </td>
                      <td className="p-4">{getStatusBadge(doc.metadata?.status as string)}</td>
                      <td className="p-4">{String(doc.metadata?.chunk_count ?? 0)}</td>
                      <td className="p-4">{formatFileSize(doc.size)}</td>
                      <td className="p-4 text-muted-foreground">{formatDate(doc.created_at)}</td>
                      <td className="p-4 text-right">
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <Trash2 className="w-4 h-4 text-destructive" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

const MAX_MEMORY_CONTENT_LENGTH = 10000;

function MemoryPage() {
  const [memories, setMemories] = useState<MemoryResult[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [newMemory, setNewMemory] = useState({
    content: "",
    category: "",
    tags: "",
    source: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleSearch = useCallback(async () => {
    // Cancel any pending search
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setLoading(true);
    try {
      const response = await searchMemories(
        {
          query: searchQuery,
          limit: 50,
        },
        abortController.signal
      );
      if (!abortController.signal.aborted) {
        setMemories(response.results);
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      console.error("Failed to search memories:", err);
      toast.error(err instanceof Error ? err.message : "Failed to search memories");
    } finally {
      if (!abortController.signal.aborted) {
        setLoading(false);
      }
    }
  }, [searchQuery]);

  useEffect(() => {
    handleSearch();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [handleSearch]);

  const validateContent = (content: string): boolean => {
    if (content.length > MAX_MEMORY_CONTENT_LENGTH) {
      setContentError(`Content exceeds maximum length of ${MAX_MEMORY_CONTENT_LENGTH} characters`);
      return false;
    }
    setContentError(null);
    return true;
  };

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setNewMemory({ ...newMemory, content: value });
    if (value.length > MAX_MEMORY_CONTENT_LENGTH) {
      setContentError(`Content exceeds maximum length of ${MAX_MEMORY_CONTENT_LENGTH} characters`);
    } else {
      setContentError(null);
    }
  };

  const handleAddMemory = async () => {
    if (!newMemory.content.trim()) return;
    if (!validateContent(newMemory.content)) return;

    setIsSubmitting(true);
    try {
      await addMemory({
        content: newMemory.content,
        category: newMemory.category || undefined,
        tags: newMemory.tags ? newMemory.tags.split(",").map((t) => t.trim()).filter(Boolean) : [],
        source: newMemory.source || undefined,
      });
      toast.success("Memory added successfully");
      // Reset form and close dialog only on success
      setNewMemory({ content: "", category: "", tags: "", source: "" });
      setContentError(null);
      setIsAddDialogOpen(false);
      handleSearch();
    } catch (err) {
      console.error("Failed to add memory:", err);
      toast.error(err instanceof Error ? err.message : "Failed to add memory");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      handleAddMemory();
    }
  };

  const handleDeleteMemory = async (id: string) => {
    if (!confirm("Are you sure you want to delete this memory?")) return;

    setIsDeleting(id);
    try {
      await deleteMemory(id);
      toast.success("Memory deleted successfully");
      handleSearch();
    } catch (err) {
      console.error("Failed to delete memory:", err);
      toast.error(err instanceof Error ? err.message : "Failed to delete memory");
    } finally {
      setIsDeleting(null);
    }
  };

  const getCategoryFromMetadata = (metadata?: Record<string, unknown>) => {
    return (metadata?.category as string) || "Uncategorized";
  };

  const getTagsFromMetadata = (metadata?: Record<string, unknown>) => {
    const tags = metadata?.tags;
    if (Array.isArray(tags)) return tags;
    return [];
  };

  const getSourceFromMetadata = (metadata?: Record<string, unknown>) => {
    return (metadata?.source as string) || "";
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Memory</h1>
          <p className="text-muted-foreground mt-1">View and manage AI memory and context</p>
        </div>
        <Button onClick={() => setIsAddDialogOpen(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Add Memory
        </Button>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search memories..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
        </div>
        <Button onClick={handleSearch} disabled={loading}>
          {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Search className="w-4 h-4 mr-2" />}
          Search
        </Button>
        <Badge variant="secondary">{memories.length} memories</Badge>
      </div>

      {isAddDialogOpen && (
        <Card>
          <CardHeader>
            <CardTitle>Add New Memory</CardTitle>
            <CardDescription>Create a new memory entry for the AI</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Content *</label>
              <Textarea
                placeholder="Enter memory content..."
                className={`min-h-[100px] ${contentError ? "border-destructive focus-visible:ring-destructive" : ""}`}
                value={newMemory.content}
                onChange={handleContentChange}
                onKeyDown={handleKeyDown}
              />
              {contentError && (
                <span className="text-xs text-destructive">{contentError}</span>
              )}
              <div className="flex justify-end">
                <span className={`text-xs ${newMemory.content.length > MAX_MEMORY_CONTENT_LENGTH ? "text-destructive" : "text-muted-foreground"}`}>
                  {newMemory.content.length}/{MAX_MEMORY_CONTENT_LENGTH}
                </span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Category</label>
                <Input
                  placeholder="e.g., facts, preferences"
                  value={newMemory.category}
                  onChange={(e) => setNewMemory({ ...newMemory, category: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Source</label>
                <Input
                  placeholder="e.g., user input, document"
                  value={newMemory.source}
                  onChange={(e) => setNewMemory({ ...newMemory, source: e.target.value })}
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Tags</label>
              <Input
                placeholder="Enter tags separated by commas..."
                value={newMemory.tags}
                onChange={(e) => setNewMemory({ ...newMemory, tags: e.target.value })}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddMemory} disabled={isSubmitting || !newMemory.content.trim()}>
                {isSubmitting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Plus className="w-4 h-4 mr-2" />}
                Add Memory
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {loading && memories.length === 0 ? (
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 space-y-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-[90%]" />
                    <Skeleton className="h-4 w-[60%]" />
                    <div className="flex flex-wrap items-center gap-2 pt-2">
                      <Skeleton className="h-5 w-[70px]" />
                      <Skeleton className="h-5 w-[50px]" />
                      <Skeleton className="h-5 w-[60px]" />
                    </div>
                  </div>
                  <Skeleton className="h-8 w-8 shrink-0" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : memories.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">
              {searchQuery ? "No memories match your search" : "No memories yet. Add some to get started."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {memories.map((memory) => (
            <Card key={memory.id}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 space-y-2">
                    <p className="text-sm whitespace-pre-wrap">{memory.content}</p>
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline">{getCategoryFromMetadata(memory.metadata)}</Badge>
                      {getTagsFromMetadata(memory.metadata).map((tag, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                      {getSourceFromMetadata(memory.metadata) && (
                        <span className="text-xs text-muted-foreground">
                          Source: {getSourceFromMetadata(memory.metadata)}
                        </span>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={() => handleDeleteMemory(memory.id)}
                    disabled={isDeleting === memory.id}
                  >
                    {isDeleting === memory.id ? (
                      <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    ) : (
                      <Trash2 className="w-4 h-4 text-destructive" />
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}



// Wrapper for SettingsPageContent that provides the health status
function SettingsPage() {
  const [health, setHealth] = useState<HealthStatus>({
    backend: false,
    embeddings: false,
    chat: false,
    loading: true,
    lastChecked: null,
  });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await getHealth();
        setHealth({
          backend: response.services?.backend ?? response.status === "ok",
          embeddings: response.services?.embeddings ?? false,
          chat: response.services?.chat ?? false,
          loading: false,
          lastChecked: new Date(),
        });
      } catch {
        setHealth({
          backend: false,
          embeddings: false,
          chat: false,
          loading: false,
          lastChecked: new Date(),
        });
      }
    };

    checkHealth();
  }, []);

  return <SettingsPageWithStatus health={health} />;
}

const pages: Record<PageId, React.ComponentType> = {
  chat: ChatPage,
  documents: DocumentsPage,
  memory: MemoryPage,
  settings: SettingsPage,
};

interface HealthStatus {
  backend: boolean;
  embeddings: boolean;
  chat: boolean;
  loading: boolean;
  lastChecked: Date | null;
}

function ConnectionStatusBadges({ health }: { health: HealthStatus }) {
  const getBadgeClass = (isUp: boolean) => {
    if (health.loading) return "bg-muted text-muted-foreground";
    return isUp ? "bg-green-500 hover:bg-green-600" : "bg-red-500 hover:bg-red-600";
  };

  const getBadgeLabel = (label: string) => {
    return health.loading ? "Checking" : label;
  };

  return (
    <div className="flex items-center gap-2">
      <Badge variant="default" className={getBadgeClass(health.backend)}>
        <Server className="w-3 h-3 mr-1" />
        {getBadgeLabel("Backend")}
      </Badge>
      <Badge variant="default" className={getBadgeClass(health.embeddings)}>
        <Cpu className="w-3 h-3 mr-1" />
        {getBadgeLabel("Embeddings")}
      </Badge>
      <Badge variant="default" className={getBadgeClass(health.chat)}>
        <MessageCircle className="w-3 h-3 mr-1" />
        {getBadgeLabel("Chat")}
      </Badge>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState<PageId>("chat");
  const [health, setHealth] = useState<HealthStatus>({
    backend: false,
    embeddings: false,
    chat: false,
    loading: true,
    lastChecked: null,
  });

  // Poll health status every 30 seconds
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await getHealth();
        setHealth({
          backend: response.services?.backend ?? response.status === "ok",
          embeddings: response.services?.embeddings ?? false,
          chat: response.services?.chat ?? false,
          loading: false,
          lastChecked: new Date(),
        });
      } catch {
        setHealth({
          backend: false,
          embeddings: false,
          chat: false,
          loading: false,
          lastChecked: new Date(),
        });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const CurrentPage = pages[activePage];

  return (
    <PageShell
      activeItem={activePage}
      onItemSelect={(id) => setActivePage(id as PageId)}
      healthStatus={health}
    >
      <CurrentPage />
    </PageShell>
  );
}

function SettingsPageWithStatus({ health }: { health: HealthStatus }) {
  const formatLastChecked = (date: Date | null) => {
    if (!date) return "Not checked";
    return `Last checked: ${date.toLocaleTimeString()}`;
  };

  const [connectionResult, setConnectionResult] = useState<ConnectionTestResult | null>(null);
  const [isTestingConnections, setIsTestingConnections] = useState(false);

  const handleConnectionTest = async () => {
    setIsTestingConnections(true);
    try {
      const result = await testConnections();
      setConnectionResult(result);
      toast.success("Connection test completed");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Connection test failed";
      toast.error(message);
      setConnectionResult(null);
    } finally {
      setIsTestingConnections(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground mt-1">Configure your application preferences</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <ConnectionStatusBadges health={health} />
          <span className="text-xs text-muted-foreground">{formatLastChecked(health.lastChecked)}</span>
          <Button size="sm" variant="outline" onClick={handleConnectionTest} disabled={isTestingConnections}>
            {isTestingConnections ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              "Test Connections"
            )}
          </Button>
          {connectionResult && (
            <div className="flex gap-2">
              {Object.entries(connectionResult).map(([service, info]) => (
                <Badge key={service} variant={info.ok ? "outline" : "destructive"} className="text-xs">
                  {service}: {info.ok ? "OK" : "Fail"}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>
      <SettingsPageContent />
    </div>
  );
}

// Rename the original SettingsPage to SettingsPageContent
function SettingsPageContent() {
  const {
    settings,
    formData,
    loading,
    saving,
    error,
    errors,
    setSettings,
    initializeForm,
    setLoading,
    setSaving,
    setError,
    updateFormField,
    validateForm,
    hasChanges,
  } = useSettingsStore();

  useEffect(() => {
    let mounted = true;
    getSettings()
      .then((data) => {
        if (mounted) {
          setSettings(data);
          initializeForm(data);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err instanceof Error ? err.message : "Failed to load settings");
          setLoading(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, [setSettings, initializeForm, setError, setLoading]);

  const handleInputChange = (field: keyof typeof formData, value: string | boolean) => {
    if (typeof value === "boolean") {
      updateFormField(field, value);
    } else {
      const numValue = parseFloat(value);
      if (!isNaN(numValue)) {
        updateFormField(field, numValue);
      }
    }
  };

  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const updated = await updateSettings({
        chunk_size: formData.chunk_size,
        chunk_overlap: formData.chunk_overlap,
        max_context_chunks: formData.max_context_chunks,
        auto_scan_enabled: formData.auto_scan_enabled,
        auto_scan_interval_minutes: formData.auto_scan_interval_minutes,
        rag_relevance_threshold: formData.rag_relevance_threshold,
      });
      setSettings(updated);
      toast.success("Settings saved successfully");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
      toast.error(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {loading && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-[180px]" />
              <Skeleton className="h-4 w-[250px]" />
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Skeleton className="h-4 w-[100px]" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-[120px]" />
                <Skeleton className="h-10 w-full" />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-[150px]" />
              <Skeleton className="h-4 w-[200px]" />
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Skeleton className="h-4 w-[80px]" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-[100px]" />
                <Skeleton className="h-10 w-full" />
              </div>
              <div className="space-y-2">
                <Skeleton className="h-4 w-[140px]" />
                <div className="flex items-center gap-4">
                  <Skeleton className="h-10 w-24" />
                  <Skeleton className="h-2 flex-1" />
                </div>
              </div>
              <div className="flex items-center gap-2 pt-4">
                <Skeleton className="h-4 w-4" />
                <Skeleton className="h-4 w-[120px]" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {error && (
        <Card>
          <CardContent className="py-8">
            <p className="text-destructive text-center">Error: {error}</p>
          </CardContent>
        </Card>
      )}

      {!loading && !error && (
        <Tabs defaultValue="general" className="w-full">
          <TabsList className="grid w-full max-w-lg grid-cols-4">
            <TabsTrigger value="general">General</TabsTrigger>
            <TabsTrigger value="ai">AI</TabsTrigger>
            <TabsTrigger value="appearance">Appearance</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>General Settings</CardTitle>
                <CardDescription>Basic application configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Application Name</label>
                  <Input
                    value={settings?.app_name || "KnowledgeVault"}
                    readOnly
                    className="bg-muted"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Default Language</label>
                  <Input
                    value={settings?.default_language || "English"}
                    readOnly
                    className="bg-muted"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="ai">
            <Card>
              <CardHeader>
                <CardTitle>AI Configuration</CardTitle>
                <CardDescription>Configure AI model and behavior</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Model</label>
                  <Input
                    value={settings?.model || "gpt-4"}
                    readOnly
                    className="bg-muted"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Temperature</label>
                  <Input
                    value={String(settings?.temperature ?? "0.7")}
                    readOnly
                    className="bg-muted"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>



          <TabsContent value="advanced" className="space-y-4">
            <div className="rounded-lg border border-amber-200 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-800 p-3 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" />
              <p className="text-sm text-amber-800 dark:text-amber-200">
                Note: Settings updates apply to the running session only.
              </p>
            </div>
            <Card>
              <CardHeader>
                <CardTitle>Advanced Settings</CardTitle>
                <CardDescription>Configure document processing and RAG parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Chunk Size */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Chunk Size</label>
                  <Input
                    type="number"
                    min={1}
                    value={formData.chunk_size}
                    onChange={(e) => handleInputChange("chunk_size", e.target.value)}
                    className={errors.chunk_size ? "border-destructive" : ""}
                  />
                  {errors.chunk_size && (
                    <p className="text-xs text-destructive">{errors.chunk_size}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Number of characters per document chunk
                  </p>
                </div>

                {/* Chunk Overlap */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Chunk Overlap</label>
                  <Input
                    type="number"
                    min={1}
                    value={formData.chunk_overlap}
                    onChange={(e) => handleInputChange("chunk_overlap", e.target.value)}
                    className={errors.chunk_overlap ? "border-destructive" : ""}
                  />
                  {errors.chunk_overlap && (
                    <p className="text-xs text-destructive">{errors.chunk_overlap}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Number of overlapping characters between chunks (must be less than chunk size)
                  </p>
                </div>

                {/* Max Context Chunks */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Max Context Chunks</label>
                  <Input
                    type="number"
                    min={1}
                    value={formData.max_context_chunks}
                    onChange={(e) => handleInputChange("max_context_chunks", e.target.value)}
                    className={errors.max_context_chunks ? "border-destructive" : ""}
                  />
                  {errors.max_context_chunks && (
                    <p className="text-xs text-destructive">{errors.max_context_chunks}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Maximum number of chunks to include in RAG context
                  </p>
                </div>

                {/* RAG Relevance Threshold */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">RAG Relevance Threshold</label>
                  <div className="flex items-center gap-4">
                    <Input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={formData.rag_relevance_threshold}
                      onChange={(e) => handleInputChange("rag_relevance_threshold", e.target.value)}
                      className={`w-24 ${errors.rag_relevance_threshold ? "border-destructive" : ""}`}
                    />
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={formData.rag_relevance_threshold}
                      onChange={(e) => handleInputChange("rag_relevance_threshold", e.target.value)}
                      className="flex-1"
                    />
                  </div>
                  {errors.rag_relevance_threshold && (
                    <p className="text-xs text-destructive">{errors.rag_relevance_threshold}</p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Minimum relevance score (0-1) for chunks to be included in context
                  </p>
                </div>

                {/* Auto Scan Enabled */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="auto_scan_enabled"
                      checked={formData.auto_scan_enabled}
                      onChange={(e) => handleInputChange("auto_scan_enabled", e.target.checked)}
                      className="h-4 w-4 rounded border-gray-300"
                    />
                    <label htmlFor="auto_scan_enabled" className="text-sm font-medium">
                      Enable Auto Scan
                    </label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Automatically scan for new documents at regular intervals
                  </p>
                </div>

                {/* Auto Scan Interval */}
                {formData.auto_scan_enabled && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Auto Scan Interval (minutes)</label>
                    <Input
                      type="number"
                      min={1}
                      value={formData.auto_scan_interval_minutes}
                      onChange={(e) => handleInputChange("auto_scan_interval_minutes", e.target.value)}
                      className={errors.auto_scan_interval_minutes ? "border-destructive" : ""}
                    />
                    {errors.auto_scan_interval_minutes && (
                      <p className="text-xs text-destructive">{errors.auto_scan_interval_minutes}</p>
                    )}
                    <p className="text-xs text-muted-foreground">
                      How often to scan for new documents (in minutes)
                    </p>
                  </div>
                )}

                {/* Save Button and Status */}
                <div className="flex items-center gap-4 pt-4 border-t">
                  <Button
                    onClick={handleSave}
                    disabled={saving || !hasChanges()}
                  >
                    {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Save Changes
                  </Button>

                  {hasChanges() && (
                    <span className="text-sm text-muted-foreground">You have unsaved changes</span>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </>
  );
}

export default App;
