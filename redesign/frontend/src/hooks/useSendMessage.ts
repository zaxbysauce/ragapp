import { useCallback, useRef } from "react";
import { useChatStore } from "@/stores/useChatStore";
import { chatStream } from "@/lib/api";

export function useSendMessage() {
  const {
    sessions,
    currentSessionId,
    isStreaming,
    abortController,
    addMessage,
    updateLastMessage,
    stopStreaming,
    createSession,
  } = useChatStore();

  const sendRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (content: string, vaultId?: string) => {
      if (isStreaming) return;

      // Create session if none exists
      let sessionId = currentSessionId;
      if (!sessionId) {
        createSession();
        sessionId = useChatStore.getState().currentSessionId;
      }
      if (!sessionId) return;

      const controller = new AbortController();
      sendRef.current = controller;

      // Add user message
      const userMessage = {
        id: crypto.randomUUID(),
        role: "user" as const,
        content,
        timestamp: Date.now(),
      };
      addMessage(sessionId, userMessage);

      // Prepare assistant placeholder
      const assistantId = crypto.randomUUID();
      const assistantMessage = {
        id: assistantId,
        role: "assistant" as const,
        content: "",
        sources: [],
        timestamp: Date.now(),
      };
      addMessage(sessionId, assistantMessage);

      // Update store with streaming state
      useChatStore.setState({
        isStreaming: true,
        abortController: controller,
      });

      try {
        await chatStream(
          {
            message: content,
            sessionId,
            vaultId,
          },
          controller.signal,
          (chunk) => {
            updateLastMessage(sessionId!, { content: chunk });
          },
          (sources) => {
            updateLastMessage(sessionId!, { sources });
          },
          (stopped) => {
            updateLastMessage(sessionId!, { stopped });
            useChatStore.setState({ isStreaming: false, abortController: null });
          },
          (error) => {
            updateLastMessage(sessionId!, { error });
            useChatStore.setState({ isStreaming: false, abortController: null });
          }
        );
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          updateLastMessage(sessionId!, { stopped: true });
        } else {
          updateLastMessage(sessionId!, { error: "Failed to send message" });
        }
        useChatStore.setState({ isStreaming: false, abortController: null });
      } finally {
        sendRef.current = null;
      }
    },
    [isStreaming, currentSessionId, addMessage, updateLastMessage, createSession]
  );

  const stopMessage = useCallback(() => {
    if (sendRef.current) {
      sendRef.current.abort();
    } else {
      stopStreaming();
    }
  }, [stopStreaming]);

  return { sendMessage, stopMessage, isStreaming };
}