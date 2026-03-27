import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, PanelRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { VaultSelector } from "@/components/vault/VaultSelector";
import { useVaultStore } from "@/stores/useVaultStore";
import { useChatStore } from "@/stores/useChatStore";
import { createChatSession } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ChatHeaderProps {
  onRightPaneToggle?: () => void;
  isRightPaneOpen?: boolean;
  className?: string;
}

export function ChatHeader({
  onRightPaneToggle,
  isRightPaneOpen = false,
  className,
}: ChatHeaderProps) {
  const navigate = useNavigate();
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { activeVaultId } = useVaultStore();
  const { newChat } = useChatStore();

  const handleVaultChange = (_vaultId: number | null) => {
    // VaultSelector already calls setActiveVault internally
    // Just clear the conversation when vault changes
    newChat();
  };

  const handleNewChat = async () => {
    setIsCreatingSession(true);
    setError(null);
    try {
      const session = await createChatSession({
        title: "New Chat",
        vault_id: activeVaultId ?? undefined,
      });
      setError(null);
      navigate(`/chat/${session.id}`);
    } catch (error) {
      console.error("Failed to create chat session:", error);
      setError("Failed to create chat session. Please try again.");
    } finally {
      setIsCreatingSession(false);
    }
  };

  return (
    <header
      className={cn(
        "relative flex items-center justify-between px-4 py-3 border-b bg-background",
        className
      )}
    >
      <div className="flex items-center gap-3">
        <VaultSelector onVaultChange={handleVaultChange} />
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={handleNewChat}
          disabled={isCreatingSession}
          className="gap-2"
        >
          <Plus className="h-4 w-4" />
          New Chat
        </Button>

        <Button
          variant="ghost"
          size="icon"
          onClick={onRightPaneToggle}
          className={cn(
            "h-9 w-9",
            isRightPaneOpen && "bg-accent text-accent-foreground"
          )}
          aria-label={isRightPaneOpen ? "Close right pane" : "Open right pane"}
          aria-pressed={isRightPaneOpen}
        >
          <PanelRight className="h-4 w-4" />
        </Button>
      </div>
      {error && (
        <p className="absolute bottom-0 left-0 right-0 text-sm text-destructive text-center py-2 bg-background border-t">
          {error}
        </p>
      )}
    </header>
  );
}
