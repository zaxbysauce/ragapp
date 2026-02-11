import { useState, useEffect, useCallback } from "react";
import { listChatSessions, getChatSession, type ChatSession } from "@/lib/api";
import { useChatStore, type Message } from "@/stores/useChatStore";

export interface UseChatHistoryReturn {
  chatHistory: ChatSession[];
  isChatLoading: boolean;
  chatHistoryError: string | null;
  handleLoadChat: (session: ChatSession) => Promise<void>;
  refreshHistory: () => Promise<void>;
}

/** Manages chat session history — fetches session list and loads individual sessions. */
export function useChatHistory(activeVaultId: number | null): UseChatHistoryReturn {
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(true);
  const [chatHistoryError, setChatHistoryError] = useState<string | null>(null);

  const refreshHistory = useCallback(async () => {
    setIsChatLoading(true);
    setChatHistoryError(null);
    try {
      const data = await listChatSessions(activeVaultId ?? undefined);
      setChatHistory(data.sessions);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load chat history";
      setChatHistoryError(errorMessage);
    } finally {
      setIsChatLoading(false);
    }
  }, [activeVaultId]);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const handleLoadChat = useCallback(async (session: ChatSession) => {
    const { isStreaming } = useChatStore.getState();
    if (isStreaming) return;
    try {
      const detail = await getChatSession(session.id);
      const loadedMessages: Message[] = detail.messages.map((m) => ({
        id: m.id.toString(),
        role: m.role as "user" | "assistant",
        content: m.content,
        sources: m.sources ?? undefined,
      }));
      useChatStore.getState().loadChat(session.id.toString(), loadedMessages);
    } catch (err) {
      console.error("Failed to load chat session:", err);
    }
  }, []);

  return {
    chatHistory,
    isChatLoading,
    chatHistoryError,
    handleLoadChat,
    refreshHistory,
  };
}
