import { useRef, useEffect } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useChatStore } from "@/stores/useChatStore";
import { cn } from "@/lib/utils";
import { MAX_INPUT_LENGTH } from "@/hooks/useSendMessage";

interface ChatInputProps {
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  className?: string;
}

export function ChatInput({ onSend, onStop, isStreaming, className }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { input, setInput, inputError } = useChatStore();

  const handleSubmit = async () => {
    if (!input.trim() || isStreaming) return;
    if (input.length > MAX_INPUT_LENGTH) return;
    onSend();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const autoResize = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  };

  useEffect(() => {
    autoResize();
  }, [input]);

  const charCount = input.length;
  const isNearLimit = charCount > MAX_INPUT_LENGTH * 0.85;
  const isOverLimit = charCount > MAX_INPUT_LENGTH;

  return (
    <div className={cn("px-4 pb-4 pt-2 shrink-0", className)}>
      <div className="max-w-3xl mx-auto">
        {/* Floating input card */}
        <div
          className={cn(
            "rounded-2xl border bg-card shadow-md transition-shadow",
            "focus-within:shadow-lg focus-within:border-primary/40",
            inputError || isOverLimit ? "border-destructive/50" : "border-border"
          )}
        >
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

          {/* Footer row */}
          <div className="flex items-center justify-between px-3 pb-2 pt-1">
            {inputError ? (
              <span className="text-xs text-destructive">{inputError}</span>
            ) : (
              <span className="text-xs text-muted-foreground/50 hidden sm:block">
                Enter to send · Shift+Enter for new line
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
