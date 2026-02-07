import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface HealthResponse {
  status: string;
  version?: string;
  timestamp?: string;
  services?: {
    backend: boolean;
    embeddings: boolean;
    chat: boolean;
  };
}

export interface ConnectionCheck {
  url: string;
  status: number | null;
  ok: boolean;
  error?: string;
}

export interface ConnectionTestResult {
  embeddings: ConnectionCheck;
  chat: ConnectionCheck;
}

export interface SettingsResponse {
  app_name?: string;
  default_language?: string;
  model?: string;
  temperature?: number;
  chunk_size?: number;
  chunk_overlap?: number;
  max_context_chunks?: number;
  auto_scan_enabled?: boolean;
  auto_scan_interval_minutes?: number;
  rag_relevance_threshold?: number;
  [key: string]: unknown;
}

export interface UpdateSettingsRequest {
  chunk_size?: number;
  chunk_overlap?: number;
  max_context_chunks?: number;
  auto_scan_enabled?: boolean;
  auto_scan_interval_minutes?: number;
  rag_relevance_threshold?: number;
}

export interface SearchMemoriesRequest {
  query: string;
  limit?: number;
  filter?: Record<string, unknown>;
}

export interface MemoryResult {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
  score?: number;
}

export interface AddMemoryRequest {
  content: string;
  category?: string;
  tags?: string[];
  source?: string;
}

export interface AddMemoryResponse {
  id: string;
  status: string;
}

export interface SearchMemoriesResponse {
  results: MemoryResult[];
  total: number;
}

export interface Document {
  id: string;
  filename: string;
  content_type?: string;
  size?: number;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

export interface ListDocumentsResponse {
  documents: Document[];
  total: number;
}

export interface UploadDocumentResponse {
  id: string;
  filename: string;
  status: string;
}

export interface DocumentStatsResponse {
  total_documents: number;
  total_chunks: number;
  total_size_bytes: number;
  documents_by_status: Record<string, number>;
}

export interface ScanDocumentsResponse {
  scanned: number;
  added: number;
  errors: string[];
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface Source {
  id: string;
  filename: string;
  snippet?: string;
  score?: number;
}

export interface ChatStreamCallbacks {
  onMessage: (chunk: string) => void;
  onSources?: (sources: Source[]) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
}

export interface ChatHistoryItem {
  id: string;
  title: string;
  lastActive: string;
  messageCount: number;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/health");
  return response.data;
}

export async function getSettings(): Promise<SettingsResponse> {
  const response = await apiClient.get<SettingsResponse>("/settings");
  return response.data;
}

export async function updateSettings(
  request: UpdateSettingsRequest
): Promise<SettingsResponse> {
  const response = await apiClient.put<SettingsResponse>("/settings", request);
  return response.data;
}

export async function testConnections(): Promise<ConnectionTestResult> {
  const response = await apiClient.get<ConnectionTestResult>("/settings/connection");
  return response.data;
}

export async function searchMemories(
  request: SearchMemoriesRequest,
  signal?: AbortSignal
): Promise<SearchMemoriesResponse> {
  try {
    const response = await apiClient.post<SearchMemoriesResponse>(
      "/memories/search",
      request,
      { signal }
    );
    return response.data;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw error;
    }
    throw new Error(
      error instanceof Error ? error.message : "Failed to search memories"
    );
  }
}

export async function addMemory(
  request: AddMemoryRequest
): Promise<AddMemoryResponse> {
  try {
    // Ensure tags is always an array, never undefined
    const payload = {
      ...request,
      tags: request.tags ?? [],
    };
    const response = await apiClient.post<AddMemoryResponse>("/memories", payload);
    return response.data;
  } catch (error) {
    throw new Error(
      error instanceof Error ? error.message : "Failed to add memory"
    );
  }
}

export async function deleteMemory(id: string): Promise<void> {
  try {
    await apiClient.delete(`/memories/${id}`);
  } catch (error) {
    throw new Error(
      error instanceof Error ? error.message : "Failed to delete memory"
    );
  }
}

export async function listDocuments(): Promise<ListDocumentsResponse> {
  const response = await apiClient.get<ListDocumentsResponse>("/documents");
  return response.data;
}

export async function uploadDocument(
  file: File,
  onProgress?: (progress: number) => void
): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post<UploadDocumentResponse>(
    "/documents",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          if (progressEvent.total) {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(progress);
          } else {
            // Total unknown - report 0 for indeterminate progress
            onProgress(0);
          }
        }
      },
    }
  );
  return response.data;
}

export async function scanDocuments(): Promise<ScanDocumentsResponse> {
  const response = await apiClient.post<ScanDocumentsResponse>("/documents/scan");
  return response.data;
}

export async function getDocumentStats(): Promise<DocumentStatsResponse> {
  const response = await apiClient.get<DocumentStatsResponse>("/documents/stats");
  return response.data;
}

export function chatStream(
  messages: ChatMessage[],
  callbacks: ChatStreamCallbacks
): () => void {
  const abortController = new AbortController();

  const startStream = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ messages }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Response body is not readable");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const data = trimmed.slice(6);
            if (data === "[DONE]") {
              callbacks.onComplete?.();
              return;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.type === 'error') {
                callbacks.onError?.(new Error(parsed.message || 'Chat stream error'));
                return;
              }
              if (parsed.content) {
                callbacks.onMessage(parsed.content);
              }
              if (parsed.sources) {
                callbacks.onSources?.(parsed.sources);
              }
            } catch {
              callbacks.onMessage(data);
            }
          }
        }
      }

      callbacks.onComplete?.();
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        return;
      }
      callbacks.onError?.(
        error instanceof Error ? error : new Error(String(error))
      );
    }
  };

  startStream();

  return () => {
    abortController.abort();
  };
}

export function getChatHistory(): ChatHistoryItem[] {
  try {
    const stored = localStorage.getItem("kv_chat_history");
    if (!stored) {
      return [];
    }
    const parsed = JSON.parse(stored);
    if (Array.isArray(parsed)) {
      return parsed as ChatHistoryItem[];
    }
    return [];
  } catch {
    return [];
  }
}

export default apiClient;
