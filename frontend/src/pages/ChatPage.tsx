import { useMemo, useState, useRef, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { MessageSquare, FileText, Plus, ChevronDown, ChevronRight, AlertCircle, BookOpen, MoreVertical, Download, RotateCcw, Edit3 } from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { type Source } from "@/lib/api";
import { useChatStore } from "@/stores/useChatStore";
import { MessageContent } from "@/components/shared/MessageContent";
import { MessageActions } from "@/components/shared/MessageActions";
import { KeyboardShortcutsDialog, useKeyboardShortcuts } from "@/components/shared/KeyboardShortcuts";
import { useVaultStore } from "@/stores/useVaultStore";
import { VaultSelector } from "@/components/vault/VaultSelector";
import { useChatHistory } from "@/hooks/useChatHistory";
import { useSendMessage, MAX_INPUT_LENGTH } from "@/hooks/useSendMessage";

export default function ChatPage() {
  const {
    messages,
    input,
    isStreaming,
    inputError,
    expandedSources,
    toggleSource,
  } = useChatStore();
  const { activeVaultId } = useVaultStore();

  const [activeTab, setActiveTab] = useState("active");
  const [showClearDialog, setShowClearDialog] = useState(false);
  const [chatTitle, setChatTitle] = useState("");
  const [isEditingTitle, setIsEditingTitle] = useState(false);

  // Keyboard shortcuts
  const { open: shortcutsOpen, setOpen: setShortcutsOpen } = useKeyboardShortcuts();

  // Chat history logic
  const { chatHistory, isChatLoading, chatHistoryError, handleLoadChat, refreshHistory } =
    useChatHistory(activeVaultId);

  // Send message logic
  const { handleSend, handleStop, handleKeyDown, handleInputChange } = useSendMessage(
    activeVaultId,
    refreshHistory
  );

  const { latestAssistantMessage, hasSources } = useMemo(() => {
    const latest = messages.filter((m) => m.role === "assistant").pop();
    return {
      latestAssistantMessage: latest,
      hasSources: !!(latest?.sources && latest.sources.length > 0),
    };
  }, [messages]);

  // Track if user is at bottom for auto-scroll
  const [isAtBottom, setIsAtBottom] = useState(true);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of messages (only if user is already at bottom)
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current && isAtBottom) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isStreaming, isAtBottom]);

  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isBottom = scrollHeight - scrollTop - clientHeight < 50;
      setIsAtBottom(isBottom);
    }
  };

  const handleToggleSource = (sourceId: string) => {
    toggleSource(sourceId);
  };

  const handleClearChat = () => {
    useChatStore.getState().clearMessages();
    setShowClearDialog(false);
  };

  const handleExportChat = () => {
    const chatText = messages.map(m => {
      const role = m.role === "user" ? "User" : "Assistant";
      return `${role}: ${m.content}`;
    }).join("\n\n");
    
    const blob = new Blob([chatText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleSaveTitle = () => {
    // TODO: Save title to backend when API is available
    setIsEditingTitle(false);
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <div>
          {isEditingTitle ? (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={chatTitle}
                onChange={(e) => setChatTitle(e.target.value)}
                placeholder="Chat title..."
                className="text-2xl font-bold bg-transparent border-b border-primary focus:outline-none focus:border-primary-foreground"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveTitle();
                  if (e.key === "Escape") setIsEditingTitle(false);
                }}
                autoFocus
              />
              <Button size="sm" onClick={handleSaveTitle}>Save</Button>
            </div>
          ) : (
            <h1 className="text-3xl font-bold tracking-tight">Chat</h1>
          )}
          <p className="text-muted-foreground mt-1">Ask questions and get AI-powered answers</p>
        </div>
        <div className="flex items-center gap-2">
          <VaultSelector />
          
          {/* Chat Actions Dropdown */}
          {messages.length > 0 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" className="min-w-[44px] min-h-[44px]" aria-label="Chat actions">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => setIsEditingTitle(true)}>
                  <Edit3 className="w-4 h-4 mr-2" />
                  Rename Chat
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleExportChat}>
                  <Download className="w-4 h-4 mr-2" />
                  Export to Text
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => setShowClearDialog(true)} className="text-destructive focus:text-destructive">
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Clear Chat
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          
          <Button variant="outline" size="sm" onClick={() => useChatStore.getState().newChat()}>
            <Plus className="w-4 h-4 mr-2" />
            New Chat
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="active">Active Chats</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="active" className="space-y-4">
          {messages.length > 0 && (
            <Card 
              ref={messagesContainerRef}
              className="min-h-[300px] max-h-[500px] overflow-y-auto"
              onScroll={handleScroll}
            >
              <CardContent 
                className="space-y-4 pt-6"
                role="log"
                aria-live="polite"
                aria-atomic="false"
                aria-label="Chat messages"
              >
                {messages.map((message, index) => (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`group max-w-[80%] rounded-lg p-4 relative ${
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
                      {/* Copy button - appears on hover/focus */}
                      <div className={`absolute -bottom-3 ${message.role === "user" ? "left-0" : "right-0"}`}>
                        <MessageActions content={message.content} />
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
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
                  aria-label="Message input"
                  aria-describedby={inputError ? "input-error" : undefined}
                />
                <div className="flex justify-between items-center">
                  {inputError ? (
                    <span id="input-error" className="text-xs text-destructive" role="alert">{inputError}</span>
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
                  {chatHistory.map((session) => (
                    <div key={session.id} className="flex items-center gap-4 p-3 rounded-lg hover:bg-muted/50 cursor-pointer transition-colors" onClick={async () => { await handleLoadChat(session); setActiveTab("active"); }}>
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <MessageSquare className="h-5 w-5 text-primary" />
                      </div>
                      <div className="flex-1">
                        <p className="font-medium">{session.title || "Untitled"}</p>
                        <p className="text-sm text-muted-foreground">Last active {new Date(session.updated_at).toLocaleString()}</p>
                      </div>
                      <span className="text-xs text-muted-foreground">{session.message_count ?? 0} messages</span>
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
          {/* Desktop Sources Panel */}
          <Card className="h-full hidden lg:flex lg:flex-col">
            <CardHeader>
              <CardTitle className="text-lg">Sources</CardTitle>
              <CardDescription>
                {hasSources
                  ? `${latestAssistantMessage!.sources!.length} source(s) for the latest response`
                  : "Sources will appear here after the AI responds"}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1">
              {hasSources ? (
                <ScrollArea className="h-[400px] pr-4">
                  <div className="space-y-3" role="list" aria-label="Document sources">
                      {latestAssistantMessage!.sources!.map((source: Source) => {
                      const isExpanded = expandedSources.has(source.id);
                      return (
                        <Card key={source.id} className="border-border/50">
                          <div
                            className="p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                            onClick={() => handleToggleSource(source.id)}
                            role="listitem"
                            aria-expanded={isExpanded}
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

          {/* Mobile Sources Accordion */}
          <div className="lg:hidden">
            {hasSources && (
              <Accordion type="single" collapsible defaultValue="sources">
                <AccordionItem value="sources" className="border rounded-lg bg-card">
                  <AccordionTrigger className="px-4 py-3 hover:no-underline">
                    <div className="flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium">
                        Sources ({latestAssistantMessage!.sources!.length})
                      </span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-4 pb-4">
                    <div className="space-y-3" role="list" aria-label="Document sources">
                      {latestAssistantMessage!.sources!.map((source: Source) => {
                        const isExpanded = expandedSources.has(source.id);
                        return (
                          <Card key={source.id} className="border-border/50">
                            <div
                              className="p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                              onClick={() => handleToggleSource(source.id)}
                              role="listitem"
                              aria-expanded={isExpanded}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <FileText className="w-4 h-4 text-muted-foreground" />
                                  <span className="text-sm font-medium truncate max-w-[200px]">
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
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}
          </div>
        </div>
      </div>

      {/* Clear Chat Confirmation Dialog */}
      <Dialog open={showClearDialog} onOpenChange={setShowClearDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clear Chat</DialogTitle>
            <DialogDescription>
              Are you sure you want to clear this conversation? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setShowClearDialog(false)}>Cancel</Button>
            <Button variant="destructive" onClick={handleClearChat}>Clear Chat</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Keyboard Shortcuts Dialog */}
      <KeyboardShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  );
}
