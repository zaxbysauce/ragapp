import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "/api";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Attach API key from localStorage if configured
apiClient.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem("kv_api_key");
  if (apiKey) {
    config.headers.Authorization = `Bearer ${apiKey}`;
  }
  return config;
});

// Normalize error responses
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Preserve AbortError for cancellation handling
    if (error.name === "AbortError" || error.code === "ERR_CANCELED") {
      return Promise.reject(error);
    }

    // Extract the most useful error message
    let message = "An unexpected error occurred";
    
    if (error.response) {
      // Server responded with an error status
      const data = error.response.data;
      message = data?.detail || data?.message || data?.error || error.response.statusText || message;
    } else if (error.request) {
      // Request was made but no response received
      message = "Unable to reach the server. Please check your connection.";
    } else {
      // Something else happened
      message = error.message || message;
    }

    // Create a normalized error with the extracted message
    const normalizedError = new Error(message);
    normalizedError.name = error.name || "APIError";
    // Preserve the original response for status code checking
    (normalizedError as any).status = error.response?.status;
    (normalizedError as any).originalError = error;
    
    return Promise.reject(normalizedError);
  }
);

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
  // Server config
  port: number;
  data_dir: string;

  // Ollama config
  ollama_embedding_url: string;
  ollama_chat_url: string;

  // Model config
  embedding_model: string;
  chat_model: string;

  // Document processing (user-configurable)
  chunk_size: number;
  chunk_overlap: number;
  max_context_chunks: number;

  // RAG config (user-configurable)
  rag_relevance_threshold: number;
  vector_top_k: number;
  retrieval_window: number;
  vector_metric: string;

  // Embedding prefixes
  embedding_doc_prefix: string;
  embedding_query_prefix: string;

  // Feature flags
  maintenance_mode: boolean;
  auto_scan_enabled: boolean;
  auto_scan_interval_minutes: number;
  enable_model_validation: boolean;

  // Limits
  max_file_size_mb: number;
  allowed_extensions: string[];

  // CORS
  backend_cors_origins: string[];
}

export interface UpdateSettingsRequest {
  chunk_size?: number;
  chunk_overlap?: number;
  max_context_chunks?: number;
  auto_scan_enabled?: boolean;
  auto_scan_interval_minutes?: number;
  rag_relevance_threshold?: number;
  retrieval_top_k?: number;
  max_distance_threshold?: number;
  retrieval_window?: number;
  vector_metric?: string;
  embedding_doc_prefix?: string;
  embedding_query_prefix?: string;
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
  messages: Array<{ id: string; role: string; content: string; sources?: Source[] }>;
}

export interface ChatSession {
  id: number;
  vault_id: number;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface ChatSessionMessage {
  id: number;
  role: string;
  content: string;
  sources: Source[] | null;
  created_at: string;
}

export interface ChatSessionDetail extends ChatSession {
  messages: ChatSessionMessage[];
}

export interface CreateSessionRequest {
  title?: string;
  vault_id?: number;
}

export interface AddMessageRequest {
  role: string;
  content: string;
  sources?: Source[];
}

export interface Vault {
  id: number;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  file_count: number;
  memory_count: number;
  session_count: number;
}

export interface VaultListResponse {
  vaults: Vault[];
}

export interface VaultCreateRequest {
  name: string;
  description?: string;
}

export interface VaultUpdateRequest {
  name?: string;
  description?: string;
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

export async function listVaults(): Promise<VaultListResponse> {
  const response = await apiClient.get<VaultListResponse>("/vaults");
  return response.data;
}

export async function getVault(id: number): Promise<Vault> {
  const response = await apiClient.get<Vault>(`/vaults/${id}`);
  return response.data;
}

export async function createVault(request: VaultCreateRequest): Promise<Vault> {
  const response = await apiClient.post<Vault>("/vaults", request);
  return response.data;
}

export async function updateVault(id: number, request: VaultUpdateRequest): Promise<Vault> {
  const response = await apiClient.put<Vault>(`/vaults/${id}`, request);
  return response.data;
}

export async function deleteVault(id: number): Promise<void> {
  await apiClient.delete(`/vaults/${id}`);
}

export async function searchMemories(
  request: SearchMemoriesRequest,
  signal?: AbortSignal,
  vaultId?: number
): Promise<SearchMemoriesResponse> {
  const body = { ...request, ...(vaultId != null && { vault_id: vaultId }) };
  const response = await apiClient.post<SearchMemoriesResponse>(
    "/memories/search",
    body,
    { signal }
  );
  return response.data;
}

export async function addMemory(
  request: AddMemoryRequest,
  vaultId?: number
): Promise<AddMemoryResponse> {
  // Ensure tags is always an array, never undefined
  const payload = {
    ...request,
    tags: request.tags ?? [],
    ...(vaultId != null && { vault_id: vaultId }),
  };
  const response = await apiClient.post<AddMemoryResponse>("/memories", payload);
  return response.data;
}

export async function deleteMemory(id: string): Promise<void> {
  await apiClient.delete(`/memories/${id}`);
}

export async function listMemories(vaultId?: number): Promise<{ memories: MemoryResult[] }> {
  const response = await apiClient.get<{ memories: MemoryResult[] }>(
    "/memories",
    vaultId != null ? { params: { vault_id: vaultId } } : undefined
  );
  return response.data;
}

export async function listDocuments(vaultId?: number): Promise<ListDocumentsResponse> {
  const response = await apiClient.get<ListDocumentsResponse>("/documents", vaultId != null ? { params: { vault_id: vaultId } } : undefined);
  return response.data;
}

export async function uploadDocument(
  file: File,
  onProgress?: (progress: number) => void,
  vaultId?: number
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
      ...(vaultId != null && { params: { vault_id: vaultId } }),
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

export async function scanDocuments(vaultId?: number): Promise<ScanDocumentsResponse> {
  const response = await apiClient.post<ScanDocumentsResponse>(
    "/documents/scan",
    undefined,
    vaultId != null ? { params: { vault_id: vaultId } } : undefined
  );
  return response.data;
}

export async function deleteDocument(fileId: string): Promise<void> {
  await apiClient.delete(`/documents/${fileId}`);
}

export async function getDocumentStats(vaultId?: number): Promise<DocumentStatsResponse> {
  const response = await apiClient.get<DocumentStatsResponse>("/documents/stats", vaultId != null ? { params: { vault_id: vaultId } } : undefined);
  return response.data;
}

export function chatStream(
  messages: ChatMessage[],
  callbacks: ChatStreamCallbacks,
  vaultId?: number
): () => void {
  const abortController = new AbortController();

  const startStream = async () => {
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      const apiKey = localStorage.getItem("kv_api_key");
      if (apiKey) {
        headers["Authorization"] = `Bearer ${apiKey}`;
      }

      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({ messages, ...(vaultId != null && { vault_id: vaultId }) }),
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

export function saveChatHistory(history: ChatHistoryItem[]): void {
  try {
    localStorage.setItem("kv_chat_history", JSON.stringify(history));
  } catch (err) {
    console.error("Failed to save chat history:", err);
  }
}

export async function listChatSessions(vaultId?: number): Promise<{ sessions: ChatSession[] }> {
  const response = await apiClient.get<{ sessions: ChatSession[] }>(
    "/chat/sessions",
    vaultId != null ? { params: { vault_id: vaultId } } : undefined
  );
  return response.data;
}

export async function getChatSession(sessionId: number): Promise<ChatSessionDetail> {
  const response = await apiClient.get<ChatSessionDetail>(`/chat/sessions/${sessionId}`);
  return response.data;
}

export async function createChatSession(request: CreateSessionRequest): Promise<ChatSession> {
  const response = await apiClient.post<ChatSession>("/chat/sessions", request);
  return response.data;
}

export async function addChatMessage(sessionId: number, request: AddMessageRequest): Promise<ChatSessionMessage> {
  const response = await apiClient.post<ChatSessionMessage>(`/chat/sessions/${sessionId}/messages`, request);
  return response.data;
}

export async function updateChatSession(sessionId: number, title: string): Promise<ChatSession> {
  const response = await apiClient.put<ChatSession>(`/chat/sessions/${sessionId}`, { title });
  return response.data;
}

export async function deleteChatSession(sessionId: number): Promise<void> {
  await apiClient.delete(`/chat/sessions/${sessionId}`);
}

export default apiClient;
