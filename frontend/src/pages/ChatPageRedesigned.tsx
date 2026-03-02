import { useEffect } from "react";
import { ChatMessages } from "@/components/chat/ChatMessages";
import { CanvasPanel } from "@/components/canvas/CanvasPanel";
import { useChatStoreRedesign } from "@/stores/useChatStoreRedesign";
import { useChatHistory } from "@/hooks/useChatHistory";
import { useVaultStore } from "@/stores/useVaultStore";
import { useKeyboardShortcuts } from "@/components/shared/KeyboardShortcuts";
import { KeyboardShortcutsDialog } from "@/components/shared/KeyboardShortcuts";
import { cn } from "@/lib/utils";

export default function ChatPageRedesigned() {
  const { canvas, setCanvasView, toggleCanvasCollapse, setCanvasWidth } = useChatStoreRedesign();
  const { activeVaultId } = useVaultStore();
  const { refreshHistory } = useChatHistory(activeVaultId);

  // Keyboard shortcuts
  const { open: shortcutsOpen, setOpen: setShortcutsOpen } = useKeyboardShortcuts();

  // Initialize chat history on mount
  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  return (
    <div className="flex h-screen w-full">
      {/* Main Chat Area */}
      <div
        className={cn(
          "flex-1 transition-all duration-300",
          !canvas.isCollapsed && "max-w-[calc(100%-300px)]"
        )}
      >
        <ChatMessages
          toggleCanvasCollapse={toggleCanvasCollapse}
          canvasCollapsed={canvas.isCollapsed}
        />
      </div>

      {/* Resizable Canvas */}
      <CanvasPanel
        canvas={canvas}
        onToggleCollapse={toggleCanvasCollapse}
        onSetView={setCanvasView}
        onSetWidth={setCanvasWidth}
      />

      {/* Keyboard Shortcuts Dialog */}
      <KeyboardShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  );
}
