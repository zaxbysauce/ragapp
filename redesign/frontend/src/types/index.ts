export interface Source {
  id: string;
  filename: string;
  snippet?: string;
  score?: number;
  content?: string;
  language?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  stopped?: boolean;
  error?: string;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export type CanvasView = "document" | "code" | null;
export type ThemeMode = "light" | "dark" | "system";

export interface CanvasState {
  view: CanvasView;
  isCollapsed: boolean;
  width: number;
  activeSourceId: string | null;
}