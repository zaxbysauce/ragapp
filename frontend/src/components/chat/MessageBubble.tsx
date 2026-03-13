import { useState } from "react";
import { motion } from "framer-motion";
import { User, Bot, Copy, Check, ThumbsUp, ThumbsDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { MessageContent } from "./MessageContent";
import type { Message } from "@/stores/useChatStore";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState<"up" | "down" | null>(null);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

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

        <MessageContent
          content={message.content}
          sources={message.sources}
          isStreaming={isStreaming && !isUser}
        />

        {message.error && (
          <div className="mt-2 text-sm text-destructive">{message.error}</div>
        )}

        {message.stopped && !message.error && (
          <div className="mt-2 text-sm text-muted-foreground italic">
            Generation stopped
          </div>
        )}

        {/* Action bar — visible on hover */}
        {!isStreaming && (
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

            {!isUser && (
              <>
                <button
                  onClick={() => setLiked(liked === "up" ? null : "up")}
                  className={cn(
                    "p-1.5 rounded-lg hover:bg-muted transition-colors",
                    liked === "up"
                      ? "text-green-500"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                  title="Good response"
                >
                  <ThumbsUp className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => setLiked(liked === "down" ? null : "down")}
                  className={cn(
                    "p-1.5 rounded-lg hover:bg-muted transition-colors",
                    liked === "down"
                      ? "text-destructive"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                  title="Bad response"
                >
                  <ThumbsDown className="h-3.5 w-3.5" />
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
