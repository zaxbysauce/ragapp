import { useState } from "react";
import { motion } from "framer-motion";
import {
  User,
  Bot,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MessageContent } from "./MessageContent";
import type { Message } from "@/stores/useChatStore";
import type { Source } from "@/lib/api";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
  onRetry?: () => void;
  onSourceClick?: (sourceId: string) => void;
}

// Parse citations like [1], [2], [3] from content
function parseCitations(content: string): Array<{ type: "text"; content: string } | { type: "citation"; index: number }> {
  const parts: Array<{ type: "text"; content: string } | { type: "citation"; index: number }> = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: "text", content: content.slice(lastIndex, match.index) });
    }
    parts.push({ type: "citation", index: parseInt(match[1], 10) });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push({ type: "text", content: content.slice(lastIndex) });
  }

  return parts.length > 0 ? parts : [{ type: "text", content }];
}

// Evidence strip component showing top 3 sources
function EvidenceStrip({
  sources,
  onSourceClick,
}: {
  sources: Source[];
  onSourceClick?: (sourceId: string) => void;
}) {
  const topSources = sources.slice(0, 3);

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {topSources.map((source, idx) => (
        <button
          key={source.id}
          onClick={() => onSourceClick?.(source.id)}
          className={cn(
            "flex items-center gap-1.5 text-xs px-2 py-1 rounded-md",
            "bg-muted/60 border border-border/60 hover:bg-muted hover:border-primary/30",
            "transition-colors"
          )}
          title={source.snippet || source.filename}
        >
          <span className="w-4 h-4 rounded-full bg-primary/10 text-primary flex items-center justify-center text-[9px] font-bold">
            {idx + 1}
          </span>
          <FileText className="h-3 w-3 text-muted-foreground" />
          <span className="truncate max-w-[120px]">{source.filename}</span>
        </button>
      ))}
      {sources.length > 3 && (
        <span className="text-xs text-muted-foreground px-1 py-1">
          +{sources.length - 3} more
        </span>
      )}
    </div>
  );
}

// Secondary actions component
function SecondaryActions({
  onCopy,
  onFeedback,
  onRetry,
  copied,
  feedback,
  showRetry,
}: {
  onCopy: () => void;
  onFeedback: (type: "up" | "down") => void;
  onRetry?: () => void;
  copied: boolean;
  feedback: "up" | "down" | null;
  showRetry: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-0.5 mt-2",
        "opacity-0 group-hover:opacity-100 transition-opacity duration-150"
      )}
    >
      <button
        onClick={onCopy}
        className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
        title="Copy message"
      >
        {copied ? (
          <Check className="h-3.5 w-3.5 text-green-500" />
        ) : (
          <Copy className="h-3.5 w-3.5" />
        )}
      </button>

      <button
        onClick={() => onFeedback("up")}
        className={cn(
          "p-1.5 rounded-lg hover:bg-muted transition-colors",
          feedback === "up"
            ? "text-green-500"
            : "text-muted-foreground hover:text-foreground"
        )}
        title="Good response"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </button>

      <button
        onClick={() => onFeedback("down")}
        className={cn(
          "p-1.5 rounded-lg hover:bg-muted transition-colors",
          feedback === "down"
            ? "text-destructive"
            : "text-muted-foreground hover:text-foreground"
        )}
        title="Bad response"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </button>

      {showRetry && onRetry && (
        <button
          onClick={onRetry}
          className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          title="Retry"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

export function MessageBubble({
  message,
  isStreaming,
  onRetry,
  onSourceClick,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = (type: "up" | "down") => {
    setFeedback(feedback === type ? null : type);
  };

  // Parse content with inline citations for assistant messages
  const parsedContent = !isUser ? parseCitations(message.content) : null;
  const hasCitations = parsedContent?.some((p) => p.type === "citation");

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={cn(
        "group relative flex gap-3 px-4 py-5",
        isUser ? "bg-primary/5" : "bg-transparent"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-0.5",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted border border-border"
        )}
      >
        {isUser ? (
          <User className="h-4 w-4" />
        ) : (
          <Bot className="h-4 w-4 text-primary" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-sm mb-1.5">
          {isUser ? "You" : "Assistant"}
        </p>

        {isUser ? (
          <MessageContent
            content={message.content}
            sources={message.sources}
            isStreaming={false}
          />
        ) : (
          <>
            {/* Layer 1: Content with inline citations */}
            <div className="prose prose-sm dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
              {hasCitations ? (
                <div className="leading-relaxed">
                  {parsedContent?.map((part, idx) =>
                    part.type === "citation" ? (
                      <span
                        key={idx}
                        className="inline-flex items-center justify-center w-5 h-5 mx-0.5 rounded bg-primary/10 text-primary text-xs font-medium cursor-pointer hover:bg-primary/20 transition-colors"
                        onClick={() => {
                          const source = message.sources?.[part.index - 1];
                          if (source) onSourceClick?.(source.id);
                        }}
                        title={`Source [${part.index}]`}
                      >
                        {part.index}
                      </span>
                    ) : (
                      <span key={idx}>{part.content}</span>
                    )
                  )}
                </div>
              ) : (
                <MessageContent
                  content={message.content}
                  sources={undefined}
                  isStreaming={isStreaming}
                />
              )}
            </div>

            {/* Layer 2: Evidence strip (top 3 sources) */}
            {message.sources && message.sources.length > 0 && !isStreaming && (
              <EvidenceStrip
                sources={message.sources}
                onSourceClick={onSourceClick}
              />
            )}
          </>
        )}

        {message.error && (
          <div className="mt-2 text-sm text-destructive">{message.error}</div>
        )}

        {message.stopped && !message.error && (
          <div className="mt-2 text-sm text-muted-foreground italic">
            Generation stopped
          </div>
        )}

        {message.interrupted && !message.stopped && !message.error && (
          <div className="mt-2 text-sm text-muted-foreground italic">
            ⚠ Response interrupted — the connection was lost mid-generation.
          </div>
        )}

        {/* Layer 3: Secondary actions */}
        {!isStreaming && !isUser && (
          <SecondaryActions
            onCopy={handleCopy}
            onFeedback={handleFeedback}
            onRetry={message.error ? onRetry : undefined}
            copied={copied}
            feedback={feedback}
            showRetry={!!message.error}
          />
        )}

        {/* User actions (just copy) */}
        {!isStreaming && isUser && (
          <div
            className={cn(
              "flex items-center gap-0.5 mt-2",
              "opacity-0 group-hover:opacity-100 transition-opacity duration-150"
            )}
          >
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
              title="Copy message"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-green-500" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}
