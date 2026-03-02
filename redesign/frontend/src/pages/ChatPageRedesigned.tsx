"use client";

import React, { useEffect } from "react";
import { ThemeProvider } from "@/components/theme-provider";
import { ChatMessages } from "@/components/chat/ChatMessages";
import { CanvasPanel } from "@/components/canvas/CanvasPanel";
import { useChatStore } from "@/stores/useChatStore";
import { cn } from "@/lib/utils";

export default function ChatPageRedesigned() {
  const { canvas } = useChatStore();

  // Apply theme class to document
  useEffect(() => {
    const interval = setInterval(() => {
      const { theme, sessions, currentSessionId, isStreaming, abortController } = useChatStore.getState();
      // Theme is handled by ThemeProvider
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <div className="flex h-screen w-full">
        {/* Main Chat Area */}
        <div
          className={cn(
            "flex-1 transition-all duration-300",
            !canvas.isCollapsed && "max-w-[calc(100%-300px)]"
          )}
        >
          <ChatMessages />
        </div>

        {/* Resizable Canvas */}
        <CanvasPanel />
      </div>
    </ThemeProvider>
  );
}