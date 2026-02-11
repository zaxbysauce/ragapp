import { useState, useEffect, useCallback } from "react";
import { getHealth } from "@/lib/api";
import type { HealthStatus } from "@/types/health";

interface UseHealthCheckOptions {
  pollInterval?: number;
}

/** Polls the backend health endpoint and returns service availability status. */
export function useHealthCheck(options?: UseHealthCheckOptions): HealthStatus {
  const [health, setHealth] = useState<HealthStatus>({
    backend: false,
    embeddings: false,
    chat: false,
    loading: true,
    lastChecked: null,
  });

  const checkHealth = useCallback(async () => {
    try {
      const response = await getHealth();
      setHealth({
        backend: response.services?.backend ?? response.status === "ok",
        embeddings: response.services?.embeddings ?? false,
        chat: response.services?.chat ?? false,
        loading: false,
        lastChecked: new Date(),
      });
    } catch {
      setHealth({
        backend: false,
        embeddings: false,
        chat: false,
        loading: false,
        lastChecked: new Date(),
      });
    }
  }, []);

  useEffect(() => {
    checkHealth();

    if (options?.pollInterval) {
      const interval = setInterval(checkHealth, options.pollInterval);
      return () => clearInterval(interval);
    }
  }, [checkHealth, options?.pollInterval]);

  return health;
}
