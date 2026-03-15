import { create } from "zustand";
import type { ChatSession } from "@/lib/api";

export type RightPaneTab = "evidence" | "preview" | "workspace";

export interface SessionGroup {
  label: string;
  sessions: ChatSession[];
}

export interface ChatShellState {
  // Session state
  sessions: ChatSession[];
  activeSessionId: string | null;
  pinnedSessionIds: Set<string>;
  sessionSearch: string;

  // Right pane state
  rightPaneOpen: boolean;
  rightPaneTab: RightPaneTab;
  focusedSourceId: string | null;

  // Loading states
  isLoadingSessions: boolean;
  sessionsError: string | null;

  // Actions
  setSessions: (sessions: ChatSession[]) => void;
  setActiveSession: (sessionId: string | null) => void;
  pinSession: (sessionId: string) => void;
  unpinSession: (sessionId: string) => void;
  setSessionSearch: (search: string) => void;

  openRightPane: (tab?: RightPaneTab) => void;
  closeRightPane: () => void;
  toggleRightPane: () => void;
  setRightPaneTab: (tab: RightPaneTab) => void;
  setFocusedSourceId: (sourceId: string | null) => void;

  setIsLoadingSessions: (loading: boolean) => void;
  setSessionsError: (error: string | null) => void;

  // Computed
  getPinnedSessions: () => ChatSession[];
  getFilteredSessions: () => ChatSession[];
  getSessionGroups: () => SessionGroup[];
}

function groupSessionsByDate(sessions: ChatSession[]): SessionGroup[] {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const lastWeek = new Date(today);
  lastWeek.setDate(lastWeek.getDate() - 7);

  const groups: Record<string, ChatSession[]> = {
    Today: [],
    Yesterday: [],
    "This Week": [],
    Older: [],
  };

  for (const session of sessions) {
    const d = new Date(session.updated_at);
    const day = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    if (day >= today) {
      groups["Today"].push(session);
    } else if (day >= yesterday) {
      groups["Yesterday"].push(session);
    } else if (day >= lastWeek) {
      groups["This Week"].push(session);
    } else {
      groups["Older"].push(session);
    }
  }

  return Object.entries(groups)
    .filter(([, s]) => s.length > 0)
    .map(([label, sessions]) => ({ label, sessions }));
}

export const useChatShellStore = create<ChatShellState>((set, get) => ({
  // Initial state
  sessions: [],
  activeSessionId: null,
  pinnedSessionIds: new Set(),
  sessionSearch: "",

  rightPaneOpen: true,
  rightPaneTab: "evidence",
  focusedSourceId: null,

  isLoadingSessions: false,
  sessionsError: null,

  // Actions
  setSessions: (sessions) => set({ sessions }),

  setActiveSession: (sessionId) => set({ activeSessionId: sessionId }),

  pinSession: (sessionId) => {
    set((state) => {
      const newPinned = new Set(state.pinnedSessionIds);
      newPinned.add(sessionId);
      return { pinnedSessionIds: newPinned };
    });
  },

  unpinSession: (sessionId) => {
    set((state) => {
      const newPinned = new Set(state.pinnedSessionIds);
      newPinned.delete(sessionId);
      return { pinnedSessionIds: newPinned };
    });
  },

  setSessionSearch: (search) => set({ sessionSearch: search }),

  openRightPane: (tab) => {
    if (tab) {
      set({ rightPaneOpen: true, rightPaneTab: tab });
    } else {
      set({ rightPaneOpen: true });
    }
  },

  closeRightPane: () => set({ rightPaneOpen: false }),

  toggleRightPane: () => set((state) => ({ rightPaneOpen: !state.rightPaneOpen })),

  setRightPaneTab: (tab) => set({ rightPaneTab: tab }),

  setFocusedSourceId: (sourceId) => set({ focusedSourceId: sourceId }),

  setIsLoadingSessions: (loading) => set({ isLoadingSessions: loading }),

  setSessionsError: (error) => set({ sessionsError: error }),

  // Computed helpers
  getPinnedSessions: () => {
    const { sessions, pinnedSessionIds } = get();
    return sessions.filter((s) => pinnedSessionIds.has(s.id.toString()));
  },

  getFilteredSessions: () => {
    const { sessions, sessionSearch } = get();
    if (!sessionSearch.trim()) return sessions;
    const q = sessionSearch.toLowerCase();
    return sessions.filter((s) =>
      (s.title || "Untitled").toLowerCase().includes(q)
    );
  },

  getSessionGroups: () => {
    const filtered = get().getFilteredSessions();
    return groupSessionsByDate(filtered);
  },
}));
