export interface SendMessageRequest {
  message: string;
  sessionId: string;
  vaultId?: string;
}

export interface StreamCallbacks {
  onChunk: (content: string) => void;
  onSources: (sources: Source[]) => void;
  onComplete: (stopped?: boolean) => void;
  onError: (error: string) => void;
}

export async function chatStream(
  req: SendMessageRequest,
  signal: AbortSignal,
  callbacks: StreamCallbacks
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No reader");

  const decoder = new TextDecoder();
  let buffer = "";
  let sourcesSent = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") {
          callbacks.onComplete();
          return;
        }
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === "chunk") {
            callbacks.onChunk(parsed.content);
          } else if (parsed.type === "sources" && !sourcesSent) {
            callbacks.onSources(parsed.sources);
            sourcesSent = true;
          } else if (parsed.type === "error") {
            callbacks.onError(parsed.message);
          }
        } catch {
          // Ignore parse errors
        }
      }
    }
  }

  callbacks.onComplete();
}