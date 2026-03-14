import { useEffect, useState } from "react";
import { ChatMessages } from "@/components/chat/ChatMessages";
import { CanvasPanel } from "@/components/canvas/CanvasPanel";
import { ConversationSidebar } from "@/components/chat/ConversationSidebar";
import { useChatStoreRedesign } from "@/stores/useChatStoreRedesign";
import { useChatHistory } from "@/hooks/useChatHistory";
import { useVaultStore } from "@/stores/useVaultStore";
import { useKeyboardShortcuts } from "@/components/shared/KeyboardShortcuts";
import { KeyboardShortcutsDialog } from "@/components/shared/KeyboardShortcuts";

export default function ChatPageRedesigned() {
  const { canvas, setCanvasView, toggleCanvasCollapse, setCanvasWidth } = useChatStoreRedesign();
  const { activeVaultId } = useVaultStore();
  const { refreshHistory } = useChatHistory(activeVaultId);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Keyboard shortcuts
  const { open: shortcutsOpen, setOpen: setShortcutsOpen } = useKeyboardShortcuts();

  // Initialize chat history on mount
  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* Left: Conversation History Sidebar */}
      <ConversationSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
      />

      {/* Center: Main Chat Area */}
      <div className="flex-1 min-w-0">
        <ChatMessages
          toggleCanvasCollapse={toggleCanvasCollapse}
          canvasCollapsed={canvas.isCollapsed}
          isSidebarOpen={isSidebarOpen}
          onToggleSidebar={() => setIsSidebarOpen((v) => !v)}
        />
      </div>

      {/* Right: Resizable Canvas */}
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
