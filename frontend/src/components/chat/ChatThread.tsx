import { useMemo } from "react";
import { useLocalRuntime, AssistantRuntimeProvider, ThreadPrimitive, MessagePrimitive, ComposerPrimitive, useMessage } from "@assistant-ui/react";
import { motion } from "framer-motion";
import { User, Bot, Send, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { RAGRuntime } from "./RAGRuntime";
import { MessageActions } from "@/components/shared/MessageActions";

// Singleton runtime instance to ensure memoization across re-renders
let runtimeInstance: RAGRuntime | null = null;

/**
 * Hook to get or create the RAGRuntime instance.
 * Ensures the same runtime is reused across component lifecycles.
 */
export function useRAGRuntime(): RAGRuntime {
  return useMemo(() => {
    if (!runtimeInstance) {
      runtimeInstance = new RAGRuntime();
    }
    return runtimeInstance;
  }, []);
}

/**
 * User message component with right-aligned layout and primary background.
 */
function UserMessage() {
  return (
    <MessagePrimitive.Root className="group relative flex gap-3 px-4 py-5 bg-primary/5">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-0.5 bg-primary text-primary-foreground order-2">
        <User className="h-4 w-4" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 order-1">
        <p className="font-semibold text-sm mb-1.5 text-right">You</p>
        <div className="text-right">
          <div className="prose prose-sm dark:prose-invert max-w-none inline-block text-left">
            <MessagePrimitive.Content />
          </div>
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

/**
 * Assistant message component with left-aligned layout and prose styling.
 */
function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="group relative flex gap-3 px-4 py-5 bg-transparent">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-0.5 bg-muted border border-border">
        <Bot className="h-4 w-4 text-primary" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-sm mb-1.5">Assistant</p>
        <div className="prose prose-sm dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
          <MessagePrimitive.Content />
        </div>
        <AssistantActions />
      </div>
    </MessagePrimitive.Root>
  );
}

/**
 * Secondary actions for assistant messages (copy, feedback).
 */
function AssistantActions() {
  const message = useMessage();
  const content = (message.content ?? [])
    .map((part: { type: string; text?: string }) => (part.type === "text" ? part.text ?? "" : ""))
    .join("");

  return (
    <div
      className={cn(
        "flex items-center gap-0.5 mt-2",
        "opacity-0 group-hover:opacity-100 transition-opacity duration-150"
      )}
    >
      <MessageActions content={content} />
    </div>
  );
}

/**
 * Streaming indicator showing a blinking cursor while assistant is generating.
 */
function StreamingIndicator() {
  return (
    <ThreadPrimitive.If running={true}>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="group relative flex gap-3 px-4 py-5 bg-transparent"
      >
        {/* Avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-0.5 bg-muted border border-border">
          <Bot className="h-4 w-4 text-primary" />
        </div>

        {/* Content with blinking cursor */}
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm mb-1.5">Assistant</p>
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Thinking</span>
            <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-1" />
          </div>
        </div>
      </motion.div>
    </ThreadPrimitive.If>
  );
}

/**
 * Composer component for message input with send button.
 */
function ChatComposer() {
  return (
    <ComposerPrimitive.Root className="px-4 pb-4 pt-2 shrink-0">
      <div className="max-w-3xl mx-auto relative">
        <div
          className={cn(
            "rounded-2xl border bg-card shadow-md transition-shadow",
            "focus-within:shadow-lg focus-within:border-primary/40",
            "border-border"
          )}
        >
          <div className="flex items-end gap-2 p-3">
            <ComposerPrimitive.Input
              placeholder="Message... (Enter to send, Shift+Enter for new line)"
              className={cn(
                "flex-1 min-h-[36px] max-h-[200px] resize-none bg-transparent",
                "focus:outline-none text-base",
                "placeholder:text-muted-foreground/50"
              )}
              rows={1}
            />
            <ComposerPrimitive.Send asChild>
              <button
                type="button"
                className={cn(
                  "h-8 w-8 rounded-xl flex items-center justify-center",
                  "bg-primary text-primary-foreground",
                  "hover:bg-primary/90 transition-colors",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </ComposerPrimitive.Send>
          </div>
        </div>
        <div className="text-xs text-muted-foreground/50 mt-2 text-center hidden sm:block">
          Enter to send · Shift+Enter for new line
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
}

/**
 * Props for the ChatThread component.
 */
interface ChatThreadProps {
  /** Optional CSS class name for styling */
  className?: string;
}

/**
 * ChatThread component using assistant-ui primitives with RAGRuntime.
 * 
 * Features:
 * - Custom user message styling (right-aligned, primary background)
 * - Custom assistant message styling (left-aligned, prose)
 * - Streaming indicator with blinking cursor
 * - Composer with send button
 * - Memoized RAGRuntime instance
 */
export function ChatThread({ className }: ChatThreadProps) {
  const runtime = useRAGRuntime();
  const threadRuntime = useLocalRuntime(runtime);

  return (
    <AssistantRuntimeProvider runtime={threadRuntime}>
      <ThreadPrimitive.Root
        className={cn("flex flex-col h-full bg-background", className)}
      >
        <ThreadPrimitive.ViewportProvider>
          {/* Messages viewport */}
          <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto w-full">
              <ThreadPrimitive.Messages
                components={{
                  UserMessage,
                  AssistantMessage,
                }}
              />
              <StreamingIndicator />
            </div>
          </ThreadPrimitive.Viewport>

          {/* Footer with composer */}
          <ThreadPrimitive.ViewportFooter>
            <ChatComposer />
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.ViewportProvider>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}
