import { useState, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Plus,
  Search,
  MessageSquare,
  Trash2,
  Edit3,
  Check,
  X,
  Pin,
  Filter,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type { ChatSession } from "@/lib/api";
import { type SessionGroup } from "@/stores/chatShellStore";

interface SessionRailProps {
  sessions: ChatSession[];
  isLoading: boolean;
  error: string | null;
  onNewChat: () => void;
  onSelectSession: (session: ChatSession) => void;
  onRenameSession: (session: ChatSession, title: string) => Promise<void>;
  onDeleteSession: (session: ChatSession) => Promise<void>;
  onPinSession: (sessionId: string) => void;
  onUnpinSession: (sessionId: string) => void;
  activeSessionId: string | null;
  pinnedSessionIds: Set<string>;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  className?: string;
}

type FilterType = "all" | "pinned" | "today" | "week";

const STORAGE_KEY = "kv_pinned_sessions";

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

// Load pinned sessions from localStorage
function loadPinnedSessions(): Set<string> {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return new Set(JSON.parse(stored));
    }
  } catch {
    // Ignore parse errors
  }
  return new Set();
}

// Save pinned sessions to localStorage
function savePinnedSessions(pinnedIds: Set<string>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...pinnedIds]));
  } catch {
    // Ignore storage errors
  }
}

interface SessionItemProps {
  session: ChatSession;
  isActive: boolean;
  isPinned: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
  onPin: () => void;
  onUnpin: () => void;
}

function SessionItem({
  session,
  isActive,
  isPinned,
  onSelect,
  onRename,
  onDelete,
  onPin,
  onUnpin,
}: SessionItemProps) {
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
    <motion.div
      layout
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -10 }}
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
          <span className="shrink-0 text-xs text-muted-foreground/60 group-hover:hidden">
            {formatRelativeTime(session.updated_at)}
          </span>
          {(showActions || isActive) && (
            <div
              className="flex items-center gap-0.5 shrink-0"
              onClick={(e) => e.stopPropagation()}
            >
              {isPinned ? (
                <button
                  onClick={onUnpin}
                  className="p-1 rounded hover:bg-muted text-primary"
                  title="Unpin"
                >
                  <Pin className="h-3 w-3 fill-current" />
                </button>
              ) : (
                <button
                  onClick={onPin}
                  className="p-1 rounded hover:bg-muted text-muted-foreground hover:text-foreground"
                  title="Pin"
                >
                  <Pin className="h-3 w-3" />
                </button>
              )}
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
    </motion.div>
  );
}

export function SessionRail({
  sessions,
  isLoading,
  error,
  onNewChat,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
  onPinSession,
  onUnpinSession,
  activeSessionId,
  pinnedSessionIds: externalPinnedIds,
  searchQuery,
  onSearchChange,
  className,
}: SessionRailProps) {
  const [filterType, setFilterType] = useState<FilterType>("all");

  // Load pinned sessions on mount
  useEffect(() => {
    const stored = loadPinnedSessions();
    if (stored.size > 0) {
      // Sync with store
      stored.forEach((id) => {
        if (!externalPinnedIds.has(id)) {
          onPinSession(id);
        }
      });
    }
  }, []);

  // Save pinned sessions when changed
  useEffect(() => {
    savePinnedSessions(externalPinnedIds);
  }, [externalPinnedIds]);

  // Get pinned sessions
  const pinnedSessions = useMemo(() => {
    return sessions.filter((s) => externalPinnedIds.has(s.id.toString()));
  }, [sessions, externalPinnedIds]);

  // Apply filters and search
  const filteredSessions = useMemo(() => {
    let result = sessions;

    // Apply search filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter((s) =>
        (s.title || "Untitled").toLowerCase().includes(q)
      );
    }

    // Apply type filter
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const lastWeek = new Date(today);
    lastWeek.setDate(lastWeek.getDate() - 7);

    switch (filterType) {
      case "pinned":
        return result.filter((s) => externalPinnedIds.has(s.id.toString()));
      case "today":
        return result.filter((s) => new Date(s.updated_at) >= today);
      case "week":
        return result.filter((s) => new Date(s.updated_at) >= lastWeek);
      default:
        return result;
    }
  }, [sessions, searchQuery, filterType, externalPinnedIds]);

  // Group sessions by date
  const groups = useMemo((): SessionGroup[] => {
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

    for (const session of filteredSessions) {
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
  }, [filteredSessions]);

  return (
    <aside
      className={cn(
        "h-full bg-card/80 border-r border-border flex flex-col overflow-hidden shrink-0 w-64",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 pt-4 pb-2">
        <span className="text-sm font-semibold text-foreground/70 tracking-wide uppercase text-[11px]">
          Sessions
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs gap-1"
          onClick={onNewChat}
        >
          <Plus className="h-3.5 w-3.5" />
          New
        </Button>
      </div>

      {/* Search with filter */}
      <div className="px-3 pb-2 space-y-2">
        <div className="flex items-center gap-2">
          <div className="flex-1 flex items-center gap-2 px-2 py-1.5 rounded-lg bg-muted/50 border border-border/60">
            <Search className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <input
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search sessions..."
              className="flex-1 min-w-0 bg-transparent text-sm focus:outline-none placeholder:text-muted-foreground/60"
            />
            {searchQuery && (
              <button onClick={() => onSearchChange("")}>
                <X className="h-3 w-3 text-muted-foreground" />
              </button>
            )}
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0">
                <Filter className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                className={cn(filterType === "all" && "bg-accent")}
                onClick={() => setFilterType("all")}
              >
                All Sessions
              </DropdownMenuItem>
              <DropdownMenuItem
                className={cn(filterType === "pinned" && "bg-accent")}
                onClick={() => setFilterType("pinned")}
              >
                Pinned Only
              </DropdownMenuItem>
              <DropdownMenuItem
                className={cn(filterType === "today" && "bg-accent")}
                onClick={() => setFilterType("today")}
              >
                Today
              </DropdownMenuItem>
              <DropdownMenuItem
                className={cn(filterType === "week" && "bg-accent")}
                onClick={() => setFilterType("week")}
              >
                This Week
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        {filterType !== "all" && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              Filter: {filterType === "pinned" ? "Pinned" : filterType === "today" ? "Today" : "This Week"}
            </span>
            <button
              onClick={() => setFilterType("all")}
              className="text-xs text-primary hover:underline"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        {isLoading ? (
          <div className="space-y-2 px-1 pt-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="flex items-center gap-2 px-2 py-1.5">
                <Skeleton className="h-4 w-4 rounded" />
                <Skeleton className="h-4 flex-1 rounded" />
              </div>
            ))}
          </div>
        ) : error ? (
          <p className="text-xs text-muted-foreground px-3 pt-4 text-center">
            Failed to load sessions
          </p>
        ) : groups.length === 0 && pinnedSessions.length === 0 ? (
          <p className="text-xs text-muted-foreground px-3 pt-4 text-center">
            {searchQuery ? "No matching sessions" : "No sessions yet"}
          </p>
        ) : (
          <AnimatePresence mode="popLayout">
            {/* Pinned sessions */}
            {pinnedSessions.length > 0 && (
              <motion.div
                key="pinned"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="mb-4"
              >
                <div className="flex items-center gap-1 px-3 py-1">
                  <Pin className="h-3 w-3 text-muted-foreground" />
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                    Pinned
                  </p>
                </div>
                {pinnedSessions.map((session) => (
                  <SessionItem
                    key={session.id}
                    session={session}
                    isActive={activeSessionId === session.id.toString()}
                    isPinned={true}
                    onSelect={() => onSelectSession(session)}
                    onRename={(title) => onRenameSession(session, title)}
                    onDelete={() => onDeleteSession(session)}
                    onPin={() => onPinSession(session.id.toString())}
                    onUnpin={() => onUnpinSession(session.id.toString())}
                  />
                ))}
              </motion.div>
            )}

            {/* Grouped sessions */}
            {groups.map(({ label, sessions }) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="mb-4"
              >
                <p className="px-3 py-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                  {label}
                </p>
                <AnimatePresence mode="popLayout">
                  {sessions.map((session) => (
                    <SessionItem
                      key={session.id}
                      session={session}
                      isActive={activeSessionId === session.id.toString()}
                      isPinned={externalPinnedIds.has(session.id.toString())}
                      onSelect={() => onSelectSession(session)}
                      onRename={(title) => onRenameSession(session, title)}
                      onDelete={() => onDeleteSession(session)}
                      onPin={() => onPinSession(session.id.toString())}
                      onUnpin={() => onUnpinSession(session.id.toString())}
                    />
                  ))}
                </AnimatePresence>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </aside>
  );
}
