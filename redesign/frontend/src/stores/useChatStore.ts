import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Message, ChatSession, CanvasState, CanvasView } from "@/types";

interface ChatStore {
  // Chat state
  sessions: ChatSession[];
  currentSessionId: string | null;
  isStreaming: boolean;
  abortController: AbortController | null;

  // Canvas state
  canvas: CanvasState;

  // Theme
  theme: ThemeMode;

  // Chat actions
  createSession: () => void;
  deleteSession: (id: string) => void;
  setCurrentSession: (id: string) => void;
  addMessage: (sessionId: string, message: Message) => void;
  updateLastMessage: (sessionId: string, updates: Partial<Message>) => void;
  stopStreaming: () => void;

  // Canvas actions
  setCanvasView: (view: CanvasView) => void;
  toggleCanvasCollapse: () => void;
  setCanvasWidth: (width: number) => void;
  setActiveSource: (sourceId: string | null) => void;

  // Theme actions
  setTheme: (theme: ThemeMode) => void;
}

const DEFAULT_CANVAS_WIDTH = 400;

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Initial state
      sessions: [],
      currentSessionId: null,
      isStreaming: false,
      abortController: null,

      canvas: {
        view: null,
        isCollapsed: true,
        width: DEFAULT_CANVAS_WIDTH,
        activeSourceId: null,
      },

      theme: "system",

      // Chat actions
      createSession: () => {
        const newSession: ChatSession = {
          id: crypto.randomUUID(),
          title: "New Chat",
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          currentSessionId: newSession.id,
        }));
      },

      deleteSession: (id) =>
        set((state) => ({
          sessions: state.sessions.filter((s) => s.id !== id),
          currentSessionId:
            state.currentSessionId === id
              ? state.sessions[0]?.id || null
              : state.currentSessionId,
        })),

      setCurrentSession: (id) => set({ currentSessionId: id }),

      addMessage: (sessionId, message) =>
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === sessionId
              ? {
                  ...session,
                  messages: [...session.messages, message],
                  updatedAt: Date.now(),
                  title:
                    session.messages.length === 0 && message.role === "user"
                      ? message.content.slice(0, 30) + "..."
                      : session.title,
                }
              : session
          ),
        })),

      updateLastMessage: (sessionId, updates) =>
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === sessionId
              ? {
                  ...session,
                  messages: session.messages.map((msg, idx) =>
                    idx === session.messages.length - 1
                      ? { ...msg, ...updates }
                      : msg
                  ),
                }
              : session
          ),
        })),

      stopStreaming: () =>
        set((state) => {
          state.abortController?.abort();
          return {
            isStreaming: false,
            abortController: null,
          };
        }),

      // Canvas actions
      setCanvasView: (view) =>
        set((state) => ({
          canvas: { ...state.canvas, view, isCollapsed: view === null },
        })),

      toggleCanvasCollapse: () =>
        set((state) => ({
          canvas: {
            ...state.canvas,
            isCollapsed: !state.canvas.isCollapsed,
          },
        })),

      setCanvasWidth: (width) =>
        set((state) => ({
          canvas: { ...state.canvas, width: Math.max(300, Math.min(800, width)) },
        })),

      setActiveSource: (sourceId) =>
        set((state) => ({
          canvas: { ...state.canvas, activeSourceId: sourceId },
        })),

      // Theme actions
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: "chat-store",
      partialize: (state) => ({
        theme: state.theme,
        canvas: {
          width: state.canvas.width,
          isCollapsed: state.canvas.isCollapsed,
        },
      }),
    }
  )
);