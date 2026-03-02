"use client";

import React, { useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./MessageBubble";
import { useChatStore } from "@/stores/useChatStore";
import { ChatInput } from "./ChatInput";
import { CanvasPanel } from "@/components/canvas/CanvasPanel";
import { Button } from "@/components/ui/button";
import { Plus, PanelLeftClose, PanelLeft } from "lucide-react";
import { cn } from "@/lib/utils";

export function ChatMessages() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const {
    currentSessionId,
    isStreaming,
    canvas,
    toggleCanvasCollapse,
    createSession,
    sessions,
  } = useChatStore();

  const currentSession = sessions.find((s) => s.id === currentSessionId);
  const messages = currentSession?.messages || [];

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className={cn("flex flex-col h-screen bg-background", canvas.isCollapsed ? "" : "pe-0")}>
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={() => createSession()}>
            <Plus className="h-4 w-4" />
          </Button>
          <h1 className="font-semibold">
            {currentSession?.title || "Chat"}
          </h1>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleCanvasCollapse}
          title={canvas.isCollapsed ? "Show canvas" : "Hide canvas"}
        >
          {canvas.isCollapsed ? (
            <PanelLeft className="h-4 w-4" />
          ) : (
            <PanelLeftClose className="h-4 w-4" />
          )}
        </Button>
      </header>

      {/* Messages */}
      <ScrollArea ref={scrollRef} className="flex-1">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center p-8">
              <div className="text-center space-y-2">
                <p className="text-lg font-medium">How can I help you today?</p>
                <p className="text-sm text-muted-foreground">
                  Ask anything. Attach documents. Get answers.
                </p>
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
      <ChatInput />
    </div>
  );
}