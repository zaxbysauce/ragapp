import { useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { useChatStore } from "@/stores/useChatStore";
import { useSendMessage } from "@/hooks/useSendMessage";
import { Button } from "@/components/ui/button";
import {
  Plus,
  PanelLeftClose,
  PanelLeft,
  Download,
  Sparkles,
  BookOpen,
  Search,
  Layers,
  HelpCircle,
} from "lucide-react";
import { VaultSelector } from "@/components/vault/VaultSelector";
import { useVaultStore } from "@/stores/useVaultStore";
import { useChatHistory } from "@/hooks/useChatHistory";
import { cn } from "@/lib/utils";

interface SuggestedPrompt {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  prompt: string;
}

const SUGGESTED_PROMPTS: SuggestedPrompt[] = [
  {
    icon: BookOpen,
    label: "Summarize documents",
    prompt: "Give me a concise summary of the key topics covered in my documents.",
  },
  {
    icon: Search,
    label: "Find key facts",
    prompt: "What are the most important facts and figures mentioned in my documents?",
  },
  {
    icon: Layers,
    label: "Compare topics",
    prompt: "Are there any conflicting or complementary ideas across my documents? Compare them.",
  },
  {
    icon: HelpCircle,
    label: "Answer a question",
    prompt: "Based on my documents, what can you tell me about ",
  },
];

interface ChatMessagesProps {
  toggleCanvasCollapse: () => void;
  canvasCollapsed: boolean;
  isSidebarOpen?: boolean;
  onToggleSidebar?: () => void;
}

export function ChatMessages({
  toggleCanvasCollapse,
  canvasCollapsed,
  isSidebarOpen,
  onToggleSidebar,
}: ChatMessagesProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { messages, isStreaming, newChat, setInput } = useChatStore();
  const { activeVaultId } = useVaultStore();
  const { refreshHistory } = useChatHistory(activeVaultId);
  const { handleSend, handleStop } = useSendMessage(activeVaultId, refreshHistory);

  const handleExportChat = () => {
    const chatText = messages
      .map((m) => {
        const role = m.role === "user" ? "User" : "Assistant";
        return `${role}: ${m.content}`;
      })
      .join("\n\n");

    const blob = new Blob([chatText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    try {
      const link = document.createElement("a");
      link.href = url;
      link.download = `chat-${new Date().toISOString().slice(0, 10)}.txt`;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();

      setTimeout(() => {
        if (document.body.contains(link)) {
          document.body.removeChild(link);
        }
        URL.revokeObjectURL(url);
      }, 100);
    } catch (error) {
      console.error("Failed to export chat:", error);
      URL.revokeObjectURL(url);
    }
  };

  const handlePromptClick = (prompt: string) => {
    setInput(prompt);
    // Auto-focus the textarea
    setTimeout(() => {
      const textarea = document.querySelector<HTMLTextAreaElement>("textarea");
      if (textarea) {
        textarea.focus();
        textarea.setSelectionRange(prompt.length, prompt.length);
      }
    }, 50);
  };

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          {onToggleSidebar && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleSidebar}
              title={isSidebarOpen ? "Hide sidebar" : "Show sidebar"}
            >
              {isSidebarOpen ? (
                <PanelLeftClose className="h-4 w-4" />
              ) : (
                <PanelLeft className="h-4 w-4" />
              )}
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={newChat} title="New chat">
            <Plus className="h-4 w-4" />
          </Button>
          <VaultSelector />
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleExportChat}
            title="Export chat"
            disabled={messages.length === 0}
          >
            <Download className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleCanvasCollapse}
            title={canvasCollapsed ? "Show canvas" : "Hide canvas"}
          >
            {canvasCollapsed ? (
              <PanelLeft className="h-4 w-4 rotate-180" />
            ) : (
              <PanelLeftClose className="h-4 w-4 rotate-180" />
            )}
          </Button>
        </div>
      </header>

      {/* Messages */}
      <ScrollArea ref={scrollRef} className="flex-1">
        <div className="max-w-3xl mx-auto w-full">
          {messages.length === 0 ? (
            /* Welcome / Empty State */
            <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 py-12 text-center">
              <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
                <Sparkles className="h-7 w-7 text-primary" />
              </div>
              <h2 className="text-2xl font-semibold tracking-tight mb-1">
                How can I help you today?
              </h2>
              <p className="text-sm text-muted-foreground mb-10 max-w-sm">
                Ask anything about your documents, or pick a suggestion below.
              </p>

              {/* Suggested prompts grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                {SUGGESTED_PROMPTS.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.label}
                      onClick={() => handlePromptClick(item.prompt)}
                      className={cn(
                        "flex items-start gap-3 p-4 rounded-xl text-left",
                        "border border-border bg-card hover:bg-muted/50",
                        "transition-colors duration-150 group"
                      )}
                    >
                      <div className="mt-0.5 p-1.5 rounded-lg bg-primary/10 text-primary shrink-0">
                        <Icon className="h-4 w-4" />
                      </div>
                      <div>
                        <p className="text-sm font-medium group-hover:text-primary transition-colors">
                          {item.label}
                        </p>
                        <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                          {item.prompt}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
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
              />
            ))
          )}
        </div>
      </ScrollArea>

      {/* Input */}
      <ChatInput onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} />
    </div>
  );
}
