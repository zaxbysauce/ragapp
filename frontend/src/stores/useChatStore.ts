import { create } from "zustand";
import type { Source } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  stopped?: boolean;
  error?: string;
}

export interface ChatState {
  // State
  messages: Message[];
  input: string;
  isStreaming: boolean;
  abortFn: (() => void) | null;
  inputError: string | null;
  expandedSources: Set<string>;

  // Actions
  setMessages: (messages: Message[] | ((prev: Message[]) => Message[])) => void;
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  clearMessages: () => void;
  setInput: (input: string) => void;
  setIsStreaming: (isStreaming: boolean) => void;
  setAbortFn: (abortFn: (() => void) | null) => void;
  setInputError: (error: string | null) => void;
  toggleSource: (sourceId: string) => void;
  clearExpandedSources: () => void;
  stopStreaming: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  // Initial state
  messages: [],
  input: "",
  isStreaming: false,
  abortFn: null,
  inputError: null,
  expandedSources: new Set(),

  // Actions
  setMessages: (messages) => {
    if (typeof messages === "function") {
      set((state) => ({ messages: messages(state.messages) }));
    } else {
      set({ messages });
    }
  },

  addMessage: (message) => {
    set((state) => ({ messages: [...state.messages, message] }));
  },

  updateMessage: (id, updates) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg
      ),
    }));
  },

  clearMessages: () => {
    set({ messages: [], expandedSources: new Set() });
  },

  setInput: (input) => {
    set({ input });
  },

  setIsStreaming: (isStreaming) => {
    set({ isStreaming });
  },

  setAbortFn: (abortFn) => {
    set({ abortFn });
  },

  setInputError: (inputError) => {
    set({ inputError });
  },

  toggleSource: (sourceId) => {
    set((state) => {
      const newSet = new Set(state.expandedSources);
      if (newSet.has(sourceId)) {
        newSet.delete(sourceId);
      } else {
        newSet.add(sourceId);
      }
      return { expandedSources: newSet };
    });
  },

  clearExpandedSources: () => {
    set({ expandedSources: new Set() });
  },

  stopStreaming: () => {
    const { abortFn } = get();
    if (abortFn) {
      abortFn();
      set({ abortFn: null, isStreaming: false });
      // Mark the last assistant message as stopped
      set((state) => {
        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          return {
            messages: state.messages.map((msg, idx) =>
              idx === state.messages.length - 1 ? { ...msg, stopped: true } : msg
            ),
          };
        }
        return state;
      });
    }
  },
}));
