import { create } from "zustand";
import { useEffect } from "react";
import { useParams } from "react-router-dom";
import apiClient, {
  type ChatSessionDetail,
  type ChatSessionMessage,
  type Source,
} from "@/lib/api";

interface ChatSessionState {
  session: ChatSessionDetail | null;
  messages: ChatSessionMessage[];
  isLoading: boolean;
  isStreaming: boolean;
  streamingContent: string;

  loadSession: (sessionId: number) => Promise<void>;
  saveMessage: (
    sessionId: number,
    role: string,
    content: string,
    sources?: Source[]
  ) => Promise<void>;
  onStreamChunk: (chunk: string) => void;
  onStreamDone: (finalContent: string, sources?: Source[]) => Promise<void>;
  clearSession: () => void;
}

const useChatSessionStore = create<ChatSessionState>((set, get) => ({
  session: null,
  messages: [],
  isLoading: false,
  isStreaming: false,
  streamingContent: "",

  loadSession: async (sessionId: number) => {
    set({ isLoading: true });
    try {
      const response = await apiClient.get<ChatSessionDetail>(
        `/chat/sessions/${sessionId}`
      );
      const session = response.data;
      if (session.id !== sessionId) {
        throw new Error("Session ID mismatch");
      }
      set({
        session,
        messages: session.messages,
        isLoading: false,
        streamingContent: "",
        isStreaming: false,
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  saveMessage: async (
    sessionId: number,
    role: string,
    content: string,
    sources?: Source[]
  ) => {
    const response = await apiClient.post<ChatSessionMessage>(
      `/chat/sessions/${sessionId}/messages`,
      {
        role,
        content,
        sources,
      }
    );
    const message = response.data;
    if (!message.id || !message.role) {
      throw new Error("Invalid message response: missing id or role");
    }
    set((state) => ({
      messages: [...state.messages, message],
    }));
  },

  onStreamChunk: (chunk: string) => {
    set((state) => ({
      isStreaming: true,
      streamingContent: state.streamingContent + chunk,
    }));
  },

  onStreamDone: async (finalContent: string, sources?: Source[]) => {
    const { session, isStreaming } = get();
    if (!isStreaming) {
      return;
    }
    if (!session) {
      set({ isStreaming: false, streamingContent: "" });
      return;
    }

    try {
      await get().saveMessage(session.id, "assistant", finalContent, sources);
      set({ isStreaming: false, streamingContent: "" });
    } catch (error) {
      set({ isStreaming: false, streamingContent: "" });
      throw error;
    }
  },

  clearSession: () => {
    set({
      session: null,
      messages: [],
      isLoading: false,
      isStreaming: false,
      streamingContent: "",
    });
  },
}));

export function useChatSession() {
  const { sessionId } = useParams<{ sessionId?: string }>();
  const store = useChatSessionStore();

  useEffect(() => {
    if (sessionId) {
      const id = parseInt(sessionId, 10);
      if (isNaN(id)) {
        console.warn("Invalid sessionId:", sessionId);
      } else {
        store.loadSession(id).catch((error) => {
          console.error("Failed to load session:", error);
        });
      }
    } else {
      store.clearSession();
    }
  }, [sessionId]);

  return store;
}

export { useChatSessionStore };
