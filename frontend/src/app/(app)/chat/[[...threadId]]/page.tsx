"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { ChatThread } from "@/components/chat/ChatThread";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { EmptyState } from "@/components/chat/EmptyState";
import { useChatSession } from "@/components/chat/useChatSession";
import { useVaultStore } from "@/stores/useVaultStore";
import { useChatStore } from "@/stores/useChatStore";
import { createChatSession } from "@/lib/api";
import { useRequirePasswordChange } from "@/lib/auth";

/**
 * Loading spinner component for the chat page.
 */
function LoadingSpinner() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
    </div>
  );
}

/**
 * Hook to check if the active vault has documents.
 */
function useVaultHasDocuments(vaultId: number | null): boolean {
  const [hasDocuments, setHasDocuments] = useState(false);
  const { getActiveVault } = useVaultStore();

  useEffect(() => {
    const checkDocuments = () => {
      if (!vaultId) {
        setHasDocuments(false);
        return;
      }
      const vault = getActiveVault();
      // Check if vault has files based on file_count property
      setHasDocuments((vault?.file_count ?? 0) > 0);
    };

    checkDocuments();
  }, [vaultId, getActiveVault]);

  return hasDocuments;
}

/**
 * Chat page component that handles:
 * - Creating new sessions at /chat
 * - Loading existing sessions at /chat/{threadId}
 * - Rendering ChatThread, ChatHeader, and EmptyState
 */
export default function ChatPage() {
  useRequirePasswordChange();
  const params = useParams<{ threadId?: string }>();
  const threadId = params.threadId;
  const navigate = useNavigate();
  const { activeVaultId, getActiveVault } = useVaultStore();
  const { isLoading, loadSession, messages, clearSession } = useChatSession();
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const loadedThreadIdRef = useRef<string | undefined>(undefined);
  const hasDocuments = useVaultHasDocuments(activeVaultId);

  const activeVault = getActiveVault();
  const vaultName = activeVault?.name ?? "Default Vault";

  // Handle new chat creation when threadId is undefined
  useEffect(() => {
    if (!threadId) {
      // Clear any existing session state
      clearSession();

      // Create a new session and navigate to it
      const handleNewChat = async () => {
        if (isCreatingSession) return;
        setIsCreatingSession(true);
        try {
          const newSession = await createChatSession({
            title: "New Chat",
            vault_id: activeVaultId ?? undefined,
          });
          navigate(`/chat/${newSession.id}`, { replace: true });
        } catch (error) {
          console.error("Failed to create chat session:", error);
          setSessionError("Failed to create chat session. Please try again.");
          setIsCreatingSession(false);
        } finally {
          setIsCreatingSession(false);
        }
      };

      handleNewChat();
    }
  }, [threadId, activeVaultId, navigate, clearSession]);

  // Load existing session when threadId is provided
  useEffect(() => {
    if (threadId && threadId !== loadedThreadIdRef.current) {
      const id = parseInt(threadId, 10);
      if (!isNaN(id)) {
        loadedThreadIdRef.current = threadId;
        loadSession(id).catch((error) => {
          const errorMessage =
            error instanceof Error ? error.message : "Failed to load chat session";
          setSessionError(`Error loading session: ${errorMessage}`);
        });
      }
    }
  }, [threadId, loadSession]);

  // Handle prompt click from EmptyState
  const handlePromptClick = (prompt: string) => {
    useChatStore.getState().setInput(prompt);
  };

  // Handle upload click from EmptyState
  const handleUploadClick = () => {
    navigate("/documents");
  };

  return (
    <div className="flex flex-col h-full">
      <ChatHeader />
      {sessionError && (
        <div className="px-4 py-2 bg-destructive/10 text-destructive text-sm text-center">
          {sessionError}
        </div>
      )}
      {isLoading || isCreatingSession ? (
        <LoadingSpinner />
      ) : messages.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <EmptyState
            vaultName={vaultName}
            hasDocuments={hasDocuments}
            onPromptClick={handlePromptClick}
            onUploadClick={handleUploadClick}
          />
        </div>
      ) : (
        <ChatThread />
      )}
    </div>
  );
}
