import { create } from "zustand";
import { persist } from "zustand/middleware";

export type CanvasView = "document" | "code" | null;

interface ChatStateRedesign {
  // Canvas state
  canvas: {
    view: CanvasView;
    isCollapsed: boolean;
    width: number;
  };
  // Theme
  theme: "light" | "dark" | "system";
  // Canvas actions
  setCanvasView: (view: CanvasView) => void;
  toggleCanvasCollapse: () => void;
  setCanvasWidth: (width: number) => void;
  // Theme actions
  setTheme: (theme: "light" | "dark" | "system") => void;
}

const DEFAULT_CANVAS_WIDTH = 400;

export const useChatStoreRedesign = create<ChatStateRedesign>()(
  persist(
    (set) => ({
      // Initial state
      canvas: {
        view: null,
        isCollapsed: true,
        width: DEFAULT_CANVAS_WIDTH,
      },
      theme: "system",

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

      // Theme actions
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: "chat-store-redesign",
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
