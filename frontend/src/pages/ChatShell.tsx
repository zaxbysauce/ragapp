import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { PanelRightOpen, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetTitle } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { useChatShellStore } from "@/stores/chatShellStore";
import { useChatStore } from "@/stores/useChatStore";
import { useVaultStore } from "@/stores/useVaultStore";
import { useChatHistory } from "@/hooks/useChatHistory";
import { useSendMessage } from "@/hooks/useSendMessage";
import { SessionRail } from "@/components/session/SessionRail";
import { RightPane } from "@/components/pane/RightPane";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { ChatInput } from "@/components/chat/ChatInput";
import { EmptyTranscript } from "@/components/chat/EmptyTranscript";
import { VaultSelector } from "@/components/vault/VaultSelector";
import {
  listChatSessions,
  getChatSession,
  updateChatSession,
  deleteChatSession,
  createChatSession,
  type ChatSession,
  type Source,
} from "@/lib/api";
import { toast } from "sonner";

// Custom hook for document stats
function useDocumentStats(vaultId: number | null) {
  const [hasDocuments, setHasDocuments] = useState(false);

  useEffect(() => {
    // Check if vault has documents
    const checkDocs = async () => {
      if (!vaultId) {
        setHasDocuments(false);
        return;
      }
      try {
        const { getDocumentStats } = await import("@/lib/api");
        const stats = await getDocumentStats(vaultId);
        setHasDocuments(stats.total_documents > 0);
      } catch {
        setHasDocuments(false);
      }
    };
    checkDocs();
  }, [vaultId]);

  return { hasDocuments };
}

// TranscriptPane - Center zone component
interface TranscriptPaneProps {
  messages: ReturnType<typeof useChatStore.getState>["messages"];
  isStreaming: boolean;
  onToggleRightPane: () => void;
  isRightPaneOpen: boolean;
  vaultSelector: React.ReactNode;
  onSourceClick?: (sourceId: string) => void;
  hasDocuments: boolean;
  onPromptClick: (prompt: string) => void;
  onUploadClick: () => void;
}

function TranscriptPane({
  messages,
  isStreaming,
  onToggleRightPane,
  isRightPaneOpen,
  vaultSelector,
  onSourceClick,
  hasDocuments,
  onPromptClick,
  onUploadClick,
}: TranscriptPaneProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { newChat } = useChatStore();

  // Auto-scroll to bottom
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
    return () => cancelAnimationFrame(id);
  }, [messages]);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          {/* Mobile menu trigger */}
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="lg:hidden h-8 w-8">
                <Menu className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="p-0 w-80">
              <SheetTitle className="sr-only">Sessions</SheetTitle>
              <MobileSessionList />
            </SheetContent>
          </Sheet>

          {vaultSelector}
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={newChat} className="hidden sm:flex">
            New Chat
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={cn("h-8 w-8 hidden lg:flex", !isRightPaneOpen && "text-muted-foreground")}
            onClick={onToggleRightPane}
            title={isRightPaneOpen ? "Hide details" : "Show details"}
          >
            <PanelRightOpen
              className={cn("h-4 w-4 transition-transform", isRightPaneOpen && "rotate-180")}
            />
          </Button>
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto w-full">
          {messages.length === 0 ? (
            <EmptyTranscript
              hasDocuments={hasDocuments}
              onPromptClick={onPromptClick}
              onUploadClick={onUploadClick}
            />
          ) : (
            messages.map((message, idx) => (
              <MessageBubble
                key={message.id}
                message={message}
                isStreaming={
                  isStreaming &&
                  idx === messages.length - 1 &&
                  message.role === "assistant"
                }
                onSourceClick={onSourceClick}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}



// Mobile session list for sheet
function MobileSessionList() {
  const navigate = useNavigate();
  const { activeVaultId } = useVaultStore();
  const {
    sessions,
    activeSessionId,
    pinnedSessionIds,
    sessionSearch,
    setSessionSearch,
    setActiveSession,
    pinSession,
    unpinSession,
  } = useChatShellStore();
  const { newChat } = useChatStore();
  const { handleLoadChat, refreshHistory } = useChatHistory(activeVaultId);

  const handleNewChat = () => {
    newChat();
    setActiveSession(null);
    navigate("/chat");
  };

  const handleSelectSession = async (session: ChatSession) => {
    await handleLoadChat(session);
    setActiveSession(session.id.toString());
    navigate(`/chat/${session.id}`);
  };

  const handleRename = async (session: ChatSession, title: string) => {
    try {
      await updateChatSession(session.id, title);
      await refreshHistory(true);
      toast.success("Session renamed");
    } catch (err) {
      toast.error("Failed to rename session");
    }
  };

  const handleDelete = async (session: ChatSession) => {
    try {
      await deleteChatSession(session.id);
      if (activeSessionId === session.id.toString()) {
        newChat();
        setActiveSession(null);
        navigate("/chat");
      }
      await refreshHistory(true);
      toast.success("Session deleted");
    } catch (err) {
      toast.error("Failed to delete session");
    }
  };

  return (
    <SessionRail
      sessions={sessions}
      isLoading={false}
      error={null}
      onNewChat={handleNewChat}
      onSelectSession={handleSelectSession}
      onRenameSession={handleRename}
      onDeleteSession={handleDelete}
      onPinSession={pinSession}
      onUnpinSession={unpinSession}
      activeSessionId={activeSessionId}
      pinnedSessionIds={pinnedSessionIds}
      searchQuery={sessionSearch}
      onSearchChange={setSessionSearch}
    />
  );
}

// Main ChatShell component
export default function ChatShell() {
  const { sessionId } = useParams<{ sessionId?: string }>();
  const navigate = useNavigate();
  const { activeVaultId } = useVaultStore();
  const { hasDocuments } = useDocumentStats(activeVaultId);
  const { setInput } = useChatStore();

  // Chat shell store
  const {
    sessions,
    setSessions,
    activeSessionId,
    setActiveSession,
    pinnedSessionIds,
    pinSession,
    unpinSession,
    sessionSearch,
    setSessionSearch,
    rightPaneOpen,
    rightPaneTab,
    focusedSourceId,
    setFocusedSourceId,
    openRightPane,
    closeRightPane,
    toggleRightPane,
    setRightPaneTab,
    setIsLoadingSessions,
    setSessionsError,
  } = useChatShellStore();

  // Chat store
  const { messages, isStreaming, newChat, loadChat } = useChatStore();

  // Hooks
  const { handleLoadChat, refreshHistory } = useChatHistory(activeVaultId);
  const { handleSend, handleStop } = useSendMessage(activeVaultId, refreshHistory);

  // Load sessions on mount
  useEffect(() => {
    const loadSessions = async () => {
      setIsLoadingSessions(true);
      setSessionsError(null);
      try {
        const data = await listChatSessions(activeVaultId ?? undefined);
        setSessions(data.sessions);
      } catch (err) {
        setSessionsError("Failed to load sessions");
        toast.error("Failed to load chat sessions");
      } finally {
        setIsLoadingSessions(false);
      }
    };

    loadSessions();
  }, [activeVaultId, setSessions, setIsLoadingSessions, setSessionsError]);

  // Handle session ID from URL
  useEffect(() => {
    if (sessionId) {
      const loadSession = async () => {
        const id = parseInt(sessionId, 10);
        if (isNaN(id)) return;

        try {
          const session = sessions.find((s) => s.id === id);
          if (session) {
            await handleLoadChat(session);
            setActiveSession(sessionId);
          } else {
            // Try to load from API
            const detail = await getChatSession(id);
            if (detail) {
              const loadedMessages = detail.messages.map((m) => ({
                id: m.id.toString(),
                role: m.role as "user" | "assistant",
                content: m.content,
                sources: m.sources ?? undefined,
              }));
              loadChat(sessionId, loadedMessages);
              setActiveSession(sessionId);
            }
          }
        } catch (err) {
          toast.error("Failed to load session");
          navigate("/chat");
        }
      };

      loadSession();
    }
  }, [sessionId, sessions, handleLoadChat, loadChat, setActiveSession, navigate]);

  // Get sources from latest assistant message
  const sources = useMemo(() => {
    const lastAssistantMessage = [...messages]
      .reverse()
      .find((m) => m.role === "assistant");
    return lastAssistantMessage?.sources;
  }, [messages]);

  // Handlers
  const handleNewChat = () => {
    newChat();
    setActiveSession(null);
    navigate("/chat");
  };

  const handleSelectSession = async (session: ChatSession) => {
    await handleLoadChat(session);
    setActiveSession(session.id.toString());
    navigate(`/chat/${session.id}`);
  };

  const handleRenameSession = async (session: ChatSession, title: string) => {
    try {
      await updateChatSession(session.id, title);
      await refreshHistory(true);
      // Refresh sessions list
      const data = await listChatSessions(activeVaultId ?? undefined);
      setSessions(data.sessions);
      toast.success("Session renamed");
    } catch (err) {
      toast.error("Failed to rename session");
    }
  };

  const handleDeleteSession = async (session: ChatSession) => {
    try {
      await deleteChatSession(session.id);
      if (activeSessionId === session.id.toString()) {
        newChat();
        setActiveSession(null);
        navigate("/chat");
      }
      // Refresh sessions list
      const data = await listChatSessions(activeVaultId ?? undefined);
      setSessions(data.sessions);
      toast.success("Session deleted");
    } catch (err) {
      toast.error("Failed to delete session");
    }
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Zone 1: SessionRail (Left) - Desktop only */}
      <div className="hidden lg:block">
        <SessionRail
          sessions={sessions}
          isLoading={false}
          error={null}
          onNewChat={handleNewChat}
          onSelectSession={handleSelectSession}
          onRenameSession={handleRenameSession}
          onDeleteSession={handleDeleteSession}
          onPinSession={pinSession}
          onUnpinSession={unpinSession}
          activeSessionId={activeSessionId}
          pinnedSessionIds={pinnedSessionIds}
          searchQuery={sessionSearch}
          onSearchChange={setSessionSearch}
        />
      </div>

      {/* Zone 2: TranscriptPane (Center) */}
      <div className="flex-1 min-w-0 flex flex-col">
        <TranscriptPane
          messages={messages}
          isStreaming={isStreaming}
          onToggleRightPane={toggleRightPane}
          isRightPaneOpen={rightPaneOpen}
          vaultSelector={<VaultSelector className="hidden sm:block" />}
          onSourceClick={(sourceId) => {
            setFocusedSourceId(sourceId);
            openRightPane("evidence");
          }}
          hasDocuments={hasDocuments}
          onPromptClick={(prompt) => {
            setInput(prompt);
            setTimeout(() => {
              const textarea = document.querySelector<HTMLTextAreaElement>("textarea");
              textarea?.focus();
            }, 50);
          }}
          onUploadClick={() => {
            // Navigate to documents or open upload modal
            navigate("/documents");
          }}
        />

        {/* Composer */}
        <ChatInput onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} />
      </div>

      {/* Zone 3: RightPane (Right) - Desktop only */}
      <div className="hidden lg:block">
        <RightPane
          isOpen={rightPaneOpen}
          activeTab={rightPaneTab}
          onTabChange={setRightPaneTab}
          onClose={closeRightPane}
          sources={sources}
          focusedSourceId={focusedSourceId}
          onFocusSource={setFocusedSourceId}
        />
      </div>

      {/* Mobile RightPane Sheet */}
      <Sheet open={rightPaneOpen} onOpenChange={(open) => !open && closeRightPane()}>
        <SheetContent side="right" className="p-0 w-full sm:w-96 lg:hidden">
          <SheetTitle className="sr-only">Details</SheetTitle>
          <div className="h-full">
            <RightPane
              isOpen={true}
              activeTab={rightPaneTab}
              onTabChange={setRightPaneTab}
              onClose={closeRightPane}
              sources={sources}
              focusedSourceId={focusedSourceId}
              onFocusSource={setFocusedSourceId}
            />
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}
