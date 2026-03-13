import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Plus,
  Search,
  MessageSquare,
  Trash2,
  Edit3,
  Check,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { useChatHistory } from "@/hooks/useChatHistory";
import { updateChatSession, deleteChatSession, type ChatSession } from "@/lib/api";
import { useChatStore } from "@/stores/useChatStore";
import { useVaultStore } from "@/stores/useVaultStore";

interface ConversationSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

function groupSessionsByDate(sessions: ChatSession[]): {
  label: string;
  sessions: ChatSession[];
}[] {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const lastWeek = new Date(today);
  lastWeek.setDate(lastWeek.getDate() - 7);

  const groups: Record<string, ChatSession[]> = {
    Today: [],
    Yesterday: [],
    "This Week": [],
    Older: [],
  };

  for (const session of sessions) {
    const d = new Date(session.updated_at);
    const day = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    if (day >= today) {
      groups["Today"].push(session);
    } else if (day >= yesterday) {
      groups["Yesterday"].push(session);
    } else if (day >= lastWeek) {
      groups["This Week"].push(session);
    } else {
      groups["Older"].push(session);
    }
  }

  return Object.entries(groups)
    .filter(([, s]) => s.length > 0)
    .map(([label, sessions]) => ({ label, sessions }));
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(minutes / 60);

  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

interface ConversationItemProps {
  session: ChatSession;
  isActive: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}

function ConversationItem({
  session,
  isActive,
  onSelect,
  onRename,
  onDelete,
}: ConversationItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(session.title || "Untitled");
  const [showActions, setShowActions] = useState(false);

  const handleRenameSubmit = () => {
    if (editTitle.trim()) {
      onRename(editTitle.trim());
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleRenameSubmit();
    if (e.key === "Escape") {
      setEditTitle(session.title || "Untitled");
      setIsEditing(false);
    }
  };

  return (
    <div
      className={cn(
        "group relative flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors text-sm",
        isActive
          ? "bg-primary/10 text-primary"
          : "text-foreground/80 hover:bg-muted/60"
      )}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
      onClick={() => !isEditing && onSelect()}
    >
      <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />

      {isEditing ? (
        <div className="flex-1 flex items-center gap-1 min-w-0" onClick={(e) => e.stopPropagation()}>
          <input
            autoFocus
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 min-w-0 bg-transparent border-b border-primary focus:outline-none text-sm"
          />
          <button
            onClick={handleRenameSubmit}
            className="shrink-0 text-green-500 hover:text-green-400"
          >
            <Check className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => {
              setEditTitle(session.title || "Untitled");
              setIsEditing(false);
            }}
            className="shrink-0 text-muted-foreground hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : (
        <>
          <span className="flex-1 truncate">{session.title || "Untitled"}</span>
          <span className="shrink-0 text-xs text-muted-foreground/60 hidden group-hover:hidden">
            {formatRelativeTime(session.updated_at)}
          </span>
          {(showActions || isActive) && (
            <div
              className="flex items-center gap-0.5 shrink-0"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => {
                  setEditTitle(session.title || "Untitled");
                  setIsEditing(true);
                }}
                className="p-1 rounded hover:bg-muted text-muted-foreground hover:text-foreground"
                title="Rename"
              >
                <Edit3 className="h-3 w-3" />
              </button>
              <button
                onClick={onDelete}
                className="p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive"
                title="Delete"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function ConversationSidebar({ isOpen, onClose: _onClose }: ConversationSidebarProps) {
  const [search, setSearch] = useState("");
  const { activeVaultId } = useVaultStore();
  const { activeChatId, newChat } = useChatStore();
  const {
    chatHistory,
    isChatLoading,
    chatHistoryError,
    handleLoadChat,
    refreshHistory,
  } = useChatHistory(activeVaultId);

  const filtered = useMemo(() => {
    if (!search.trim()) return chatHistory;
    const q = search.toLowerCase();
    return chatHistory.filter((s) =>
      (s.title || "Untitled").toLowerCase().includes(q)
    );
  }, [chatHistory, search]);

  const groups = useMemo(() => groupSessionsByDate(filtered), [filtered]);

  const handleRename = async (session: ChatSession, title: string) => {
    try {
      await updateChatSession(session.id, title);
      await refreshHistory(true);
    } catch (err) {
      console.error("Failed to rename session:", err);
    }
  };

  const handleDelete = async (session: ChatSession) => {
    try {
      await deleteChatSession(session.id);
      // If we deleted the active chat, start fresh
      if (activeChatId === session.id.toString()) {
        newChat();
      }
      await refreshHistory(true);
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.aside
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 260, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.25, ease: "easeInOut" }}
          className="relative h-full bg-card/80 border-r border-border flex flex-col overflow-hidden shrink-0"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-3 pt-4 pb-2">
            <span className="text-sm font-semibold text-foreground/70 tracking-wide uppercase text-[11px]">
              Conversations
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs gap-1"
              onClick={newChat}
            >
              <Plus className="h-3.5 w-3.5" />
              New
            </Button>
          </div>

          {/* Search */}
          <div className="px-3 pb-2">
            <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-muted/50 border border-border/60">
              <Search className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search conversations..."
                className="flex-1 min-w-0 bg-transparent text-sm focus:outline-none placeholder:text-muted-foreground/60"
              />
              {search && (
                <button onClick={() => setSearch("")}>
                  <X className="h-3 w-3 text-muted-foreground" />
                </button>
              )}
            </div>
          </div>

          {/* Conversation list */}
          <div className="flex-1 overflow-y-auto px-2 pb-4">
            {isChatLoading ? (
              <div className="space-y-2 px-1 pt-2">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="flex items-center gap-2 px-2 py-1.5">
                    <Skeleton className="h-4 w-4 rounded" />
                    <Skeleton className="h-4 flex-1 rounded" />
                  </div>
                ))}
              </div>
            ) : chatHistoryError ? (
              <p className="text-xs text-muted-foreground px-3 pt-4 text-center">
                Failed to load history
              </p>
            ) : groups.length === 0 ? (
              <p className="text-xs text-muted-foreground px-3 pt-4 text-center">
                {search ? "No matching conversations" : "No conversations yet"}
              </p>
            ) : (
              groups.map(({ label, sessions }) => (
                <div key={label} className="mb-4">
                  <p className="px-3 py-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                    {label}
                  </p>
                  {sessions.map((session) => (
                    <ConversationItem
                      key={session.id}
                      session={session}
                      isActive={activeChatId === session.id.toString()}
                      onSelect={() => handleLoadChat(session)}
                      onRename={(title) => handleRename(session, title)}
                      onDelete={() => handleDelete(session)}
                    />
                  ))}
                </div>
              ))
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
