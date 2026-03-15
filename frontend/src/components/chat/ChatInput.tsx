import { useRef, useEffect, useState, useCallback } from "react";
import { Send, Square, Sparkles, FileText, GitCompare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useChatStore } from "@/stores/useChatStore";
import { cn } from "@/lib/utils";
import { MAX_INPUT_LENGTH } from "@/hooks/useSendMessage";

interface ChatInputProps {
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  isProcessing?: boolean;
  className?: string;
}

interface SlashCommand {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  template: string;
}

const SLASH_COMMANDS: SlashCommand[] = [
  {
    id: "summarize",
    label: "Summarize",
    description: "Create a concise summary of documents",
    icon: FileText,
    template: "Summarize the key points from my documents about ",
  },
  {
    id: "compare",
    label: "Compare",
    description: "Compare two or more topics or documents",
    icon: GitCompare,
    template: "Compare and contrast the following: ",
  },
  {
    id: "analyze",
    label: "Analyze",
    description: "Deep analysis with key insights",
    icon: Sparkles,
    template: "Analyze the following and provide key insights: ",
  },
];

// Shimmer placeholder animation component
function ShimmerPlaceholder() {
  return (
    <div className="absolute inset-0 flex items-center px-4 pointer-events-none overflow-hidden">
      <div className="flex items-center gap-2 w-full">
        <div className="h-4 w-4 rounded-full bg-muted animate-pulse" />
        <div className="h-4 flex-1 max-w-[200px] bg-gradient-to-r from-muted via-muted/50 to-muted bg-[length:200%_100%] animate-shimmer rounded" />
      </div>
    </div>
  );
}

// Slash commands menu
function SlashCommandsMenu({
  isOpen,
  onSelect,
  searchQuery,
}: {
  isOpen: boolean;
  onSelect: (command: SlashCommand) => void;
  searchQuery: string;
}) {
  const filtered = searchQuery
    ? SLASH_COMMANDS.filter(
        (c) =>
          c.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
          c.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : SLASH_COMMANDS;

  if (!isOpen || filtered.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="absolute bottom-full left-0 right-0 mb-2 bg-card border border-border rounded-xl shadow-lg overflow-hidden z-50"
    >
      <div className="p-2">
        <p className="text-xs text-muted-foreground px-2 py-1">Commands</p>
        {filtered.map((command) => {
          const Icon = command.icon;
          return (
            <button
              key={command.id}
              onClick={() => onSelect(command)}
              className="w-full flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-muted transition-colors text-left"
            >
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <Icon className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">/{command.label}</p>
                <p className="text-xs text-muted-foreground truncate">
                  {command.description}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </motion.div>
  );
}

export function ChatInput({
  onSend,
  onStop,
  isStreaming,
  isProcessing,
  className,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { input, setInput, inputError } = useChatStore();
  const [showSlashMenu, setShowSlashMenu] = useState(false);
  const [slashSearch, setSlashSearch] = useState("");

  const handleSubmit = useCallback(async () => {
    if (!input.trim() || isStreaming) return;
    if (input.length > MAX_INPUT_LENGTH) return;
    onSend();
  }, [input, isStreaming, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (showSlashMenu) {
          // Select first command if menu is open
          const filtered = slashSearch
            ? SLASH_COMMANDS.filter((c) =>
                c.label.toLowerCase().includes(slashSearch.toLowerCase())
              )
            : SLASH_COMMANDS;
          if (filtered.length > 0) {
            handleSlashSelect(filtered[0]);
          }
        } else {
          handleSubmit();
        }
      }
      if (e.key === "Escape") {
        setShowSlashMenu(false);
      }
    },
    [showSlashMenu, slashSearch, handleSubmit]
  );

  const handleSlashSelect = useCallback(
    (command: SlashCommand) => {
      setInput(command.template);
      setShowSlashMenu(false);
      setSlashSearch("");
      // Focus and move cursor to end
      setTimeout(() => {
        textareaRef.current?.focus();
        const len = command.template.length;
        textareaRef.current?.setSelectionRange(len, len);
      }, 0);
    },
    [setInput]
  );

  const autoResize = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, []);

  useEffect(() => {
    autoResize();
  }, [input, autoResize]);

  // Detect slash command trigger
  useEffect(() => {
    const trimmed = input.trimStart();
    if (trimmed.startsWith("/")) {
      const search = trimmed.slice(1).split(/\s/)[0];
      setSlashSearch(search);
      setShowSlashMenu(true);
    } else {
      setShowSlashMenu(false);
      setSlashSearch("");
    }
  }, [input]);

  const charCount = input.length;
  const isNearLimit = charCount > MAX_INPUT_LENGTH * 0.85;
  const isOverLimit = charCount > MAX_INPUT_LENGTH;

  return (
    <div className={cn("px-4 pb-4 pt-2 shrink-0", className)}>
      <div className="max-w-3xl mx-auto relative">
        {/* Pre-allocated assistant row with shimmer */}
        <AnimatePresence>
          {(isStreaming || isProcessing) && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-3 flex items-center gap-2 px-1"
            >
              <div className="w-6 h-6 rounded-full bg-muted border border-border flex items-center justify-center">
                <Sparkles className="h-3 w-3 text-primary animate-pulse" />
              </div>
              <div className="flex-1 h-8 bg-gradient-to-r from-muted/50 via-muted to-muted/50 bg-[length:200%_100%] animate-shimmer rounded-lg" />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Slash commands menu */}
        <AnimatePresence>
          {showSlashMenu && !isStreaming && (
            <SlashCommandsMenu
              isOpen={showSlashMenu}
              onSelect={handleSlashSelect}
              searchQuery={slashSearch}
            />
          )}
        </AnimatePresence>

        {/* Floating input card */}
        <div
          className={cn(
            "rounded-2xl border bg-card shadow-md transition-shadow",
            "focus-within:shadow-lg focus-within:border-primary/40",
            inputError || isOverLimit ? "border-destructive/50" : "border-border"
          )}
        >
          <div className="relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
              }}
              placeholder="Message… (Enter to send, Shift+Enter for new line)"
              className={cn(
                "min-h-[52px] max-h-[200px] resize-none border-0 shadow-none bg-transparent",
                "focus-visible:ring-0 focus-visible:ring-offset-0",
                "text-base px-4 pt-3 pb-1 rounded-2xl"
              )}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
            />
            {/* Shimmer placeholder when streaming */}
            {isStreaming && <ShimmerPlaceholder />}
          </div>

          {/* Footer row */}
          <div className="flex items-center justify-between px-3 pb-2 pt-1">
            {inputError ? (
              <span className="text-xs text-destructive">{inputError}</span>
            ) : (
              <span className="text-xs text-muted-foreground/50 hidden sm:block">
                {showSlashMenu
                  ? "↑↓ to navigate · Enter to select"
                  : "Enter to send · Shift+Enter for new line · / for commands"}
              </span>
            )}

            <div className="flex items-center gap-2 ml-auto">
              {/* Char counter */}
              {isNearLimit && (
                <span
                  className={cn(
                    "text-xs tabular-nums",
                    isOverLimit ? "text-destructive" : "text-muted-foreground"
                  )}
                >
                  {charCount}/{MAX_INPUT_LENGTH}
                </span>
              )}

              {/* Send / Stop button */}
              {isStreaming ? (
                <Button
                  variant="destructive"
                  size="icon"
                  className="h-8 w-8 rounded-xl"
                  onClick={onStop}
                >
                  <Square className="h-3.5 w-3.5" />
                </Button>
              ) : (
                <Button
                  size="icon"
                  className="h-8 w-8 rounded-xl"
                  onClick={handleSubmit}
                  disabled={!input.trim() || isOverLimit}
                >
                  <Send className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
