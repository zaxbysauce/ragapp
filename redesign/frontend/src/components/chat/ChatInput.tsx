"use client";

import React, { useRef, useEffect } from "react";
import { Send, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useSendMessage } from "@/hooks/useSendMessage";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  className?: string;
}

export function ChatInput({ className }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { sendMessage, stopMessage, isStreaming } = useSendMessage();

  const handleSubmit = async () => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const content = textarea.value.trim();
    if (!content) return;

    textarea.value = "";
    textarea.style.height = "auto";

    await sendMessage(content);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  };

  useEffect(() => {
    handleInput();
  }, []);

  return (
    <div className={cn("flex items-end gap-2 p-4 border-t border-border", className)}>
      <Textarea
        ref={textareaRef}
        placeholder="Message... (Enter to send, Shift+Enter for new line)"
        className="min-h-[44px] max-h-32 resize-none"
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        disabled={isStreaming}
      />
      {isStreaming ? (
        <Button variant="destructive" size="icon" onClick={stopMessage}>
          <Square className="h-4 w-4" />
        </Button>
      ) : (
        <Button size="icon" onClick={handleSubmit}>
          <Send className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}