import { apiStream } from "@/lib/api";
import type { Source } from "@/lib/api";
import type {
  ChatModelAdapter,
  ChatModelRunOptions,
  ChatModelRunResult,
  ThreadMessage,
  TextMessagePart,
  ThreadAssistantMessagePart,
} from "@assistant-ui/react";

/**
 * Message format expected by the backend API.
 */
interface ApiMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

/**
 * Model context containing vaultId for RAG operations.
 */
interface RAGModelContext {
  vaultId?: number;
}

/**
 * RAGRuntime implements ChatModelAdapter to stream chat responses
 * from the backend with support for vault-specific document retrieval.
 */
export class RAGRuntime implements ChatModelAdapter {
  private abortFn: (() => void) | null = null;

  /**
   * Converts ThreadMessage array to the API message format.
   * Flattens message content parts into a single string.
   */
  private convertMessages(messages: readonly ThreadMessage[]): ApiMessage[] {
    return messages.map((message) => {
      // Handle content based on message role and content structure
      let contentText = "";

      if (Array.isArray(message.content)) {
        // Flatten content parts - for user messages, parts have 'text' property
        // For assistant messages, parts can have 'text' (TextMessagePart) or other types
        contentText = message.content
          .map((part) => {
            // TextMessagePart has 'text' property
            if ("text" in part && typeof part.text === "string") {
              return part.text;
            }
            // Fallback: try to get content if it exists
            if ("content" in part && typeof part.content === "string") {
              return part.content;
            }
            return "";
          })
          .join("");
      } else if (typeof message.content === "string") {
        contentText = message.content;
      }

      return {
        role: message.role,
        content: contentText,
      };
    });
  }

  /**
   * Creates a text message part for the content array.
   */
  private createTextPart(text: string): TextMessagePart {
    return {
      type: "text",
      text,
    };
  }

  /**
   * Creates a streaming update result with the accumulated content.
   */
  private createStreamingUpdate(content: string): ChatModelRunResult {
    return {
      content: [this.createTextPart(content)],
      status: { type: "running" },
    };
  }

  /**
   * Creates the final result with sources metadata.
   */
  private createFinalResult(
    content: string,
    sources: Source[] | undefined
  ): ChatModelRunResult {
    const contentParts: ThreadAssistantMessagePart[] = [
      this.createTextPart(content),
    ];

    // Add source parts if available
    if (sources && sources.length > 0) {
      for (const source of sources) {
        contentParts.push({
          type: "source",
          sourceType: "url",
          id: source.id,
          url: `#source-${source.id}`,
          title: source.filename,
        });
      }
    }

    return {
      content: contentParts,
      status: {
        type: "complete",
        reason: "stop",
      },
    };
  }

  async *run(
    options: ChatModelRunOptions
  ): AsyncGenerator<ChatModelRunResult, void> {
    const { messages, abortSignal, context } = options;

    // Convert ThreadMessage[] to API format
    const apiMessages = this.convertMessages(messages);

    // Extract vault_id from context
    const ragContext = context as RAGModelContext | undefined;
    const vaultId = ragContext?.vaultId ?? null;

    // Track accumulated content and sources
    let accumulatedContent = "";
    let finalSources: Source[] | undefined;
    let streamError: Error | null = null;
    let isComplete = false;

    // Start the stream and capture abort function immediately
    const stream = apiStream(
      "/chat/stream",
      {
        messages: apiMessages,
        ...(vaultId != null && { vault_id: vaultId }),
      },
      {
        onChunk: (text: string) => {
          accumulatedContent += text;
        },
        onDone: (sources) => {
          finalSources = sources;
          isComplete = true;
        },
        onError: (error: Error) => {
          streamError = error;
        },
      }
    );

    // Capture abort function before any abort checks
    this.abortFn = stream.abort;

    // Create a promise that resolves when streaming is done or errors
    const streamPromise = new Promise<void>((resolve, reject) => {
      // Poll for completion by checking the stream callbacks' side effects
      const checkComplete = () => {
        if (streamError) {
          reject(streamError);
        } else if (isComplete) {
          resolve();
        } else {
          setTimeout(checkComplete, 10);
        }
      };
      checkComplete();
    });

    // Set up abort handling with try/finally
    let handleAbort: (() => void) | undefined;
    if (abortSignal) {
      handleAbort = () => {
        this.abort();
      };

      // Check if already aborted before attaching listener
      if (abortSignal.aborted) {
        handleAbort();
      } else {
        abortSignal.addEventListener("abort", handleAbort);
      }
    }

    try {
      // Set up polling mechanism to yield updates while streaming
      let lastYieldedContent = "";

      // Yield updates while waiting for the stream to complete
      while (!isComplete && !streamError) {
        // Check if there's new content to yield
        if (accumulatedContent.length > lastYieldedContent.length) {
          lastYieldedContent = accumulatedContent;
          yield this.createStreamingUpdate(accumulatedContent);
        }

        // Small delay to prevent tight polling loop
        await new Promise((resolve) => setTimeout(resolve, 50));

        // Check if stream completed during the delay
        if (isComplete) break;
      }

      // Wait for the stream to fully complete (or error)
      try {
        await streamPromise;
      } catch (error) {
        // Error is already captured in streamError
        if (streamError) {
          throw streamError;
        }
        throw error;
      }

      // Yield final result
      yield this.createFinalResult(accumulatedContent, finalSources);
    } finally {
      // Clean up ALWAYS - even if consumer abandons the generator
      if (abortSignal && handleAbort) {
        abortSignal.removeEventListener("abort", handleAbort);
      }
      if (this.abortFn) {
        this.abortFn();
        this.abortFn = null;
      }
    }
  }

  /**
   * Aborts the ongoing stream if one is active.
   */
  abort(): void {
    if (this.abortFn) {
      this.abortFn();
      this.abortFn = null;
    }
  }
}
