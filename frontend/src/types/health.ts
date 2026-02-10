export interface HealthStatus {
  backend: boolean;
  embeddings: boolean;
  chat: boolean;
  loading: boolean;
  lastChecked: Date | null;
}
