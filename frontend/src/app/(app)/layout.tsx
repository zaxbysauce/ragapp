import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { SessionRail } from "@/components/session/SessionRail";
import { useChatShellStore } from "@/stores/chatShellStore";
import { useAuthStore } from "@/stores/authStore";
import { listChatSessions, createChatSession, updateChatSession, deleteChatSession } from "@/lib/api";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ChatSession } from "@/lib/api";

interface AppLayoutProps {
  children: React.ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const navigate = useNavigate();
  const authStore = useAuthStore();
  const chatShellStore = useChatShellStore();

  const {
    sessions,
    isLoadingSessions,
    sessionsError,
    activeSessionId,
    pinnedSessionIds,
    sessionSearch,
    setSessions,
    setActiveSession,
    setSessionSearch,
    pinSession,
    unpinSession,
    setIsLoadingSessions,
    setSessionsError,
  } = chatShellStore;

  const mustChangePassword = authStore.mustChangePassword;

  // Check auth on mount
  useEffect(() => {
    authStore.checkAuth();
  }, []);

  // Fetch sessions on mount
  useEffect(() => {
    const fetchSessions = async () => {
      setIsLoadingSessions(true);
      setSessionsError(null);
      try {
        const response = await listChatSessions();
        setSessions(response.sessions);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to load sessions";
        setSessionsError(message);
      } finally {
        setIsLoadingSessions(false);
      }
    };

    fetchSessions();
  }, [setSessions, setIsLoadingSessions, setSessionsError]);

  const handleNewChat = async () => {
    try {
      const session = await createChatSession({ title: "New Chat" });
      setSessions([session, ...sessions]);
      setActiveSession(session.id.toString());
      navigate(`/chat/${session.id}`);
    } catch (error) {
      console.error("Failed to create chat session:", error);
    }
  };

  const handleSelectSession = (session: ChatSession) => {
    setActiveSession(session.id.toString());
    navigate(`/chat/${session.id}`);
  };

  const handleRenameSession = async (session: ChatSession, title: string) => {
    try {
      await updateChatSession(session.id, title);
      const updatedSessions = sessions.map((s) =>
        s.id === session.id ? { ...s, title } : s
      );
      setSessions(updatedSessions);
    } catch (error) {
      console.error("Failed to rename session:", error);
    }
  };

  const handleDeleteSession = async (session: ChatSession) => {
    try {
      await deleteChatSession(session.id);
      const filteredSessions = sessions.filter((s) => s.id !== session.id);
      setSessions(filteredSessions);
      if (activeSessionId === session.id.toString()) {
        setActiveSession(null);
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
    }
  };

  const handlePinSession = (sessionId: string) => {
    pinSession(sessionId);
  };

  const handleUnpinSession = (sessionId: string) => {
    unpinSession(sessionId);
  };

  const handleSearchChange = (query: string) => {
    setSessionSearch(query);
  };

  return (
    <div className="flex h-screen">
      {/* Left: SessionRail */}
      <SessionRail
        sessions={sessions}
        isLoading={isLoadingSessions}
        error={sessionsError}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
        onPinSession={handlePinSession}
        onUnpinSession={handleUnpinSession}
        activeSessionId={activeSessionId}
        pinnedSessionIds={pinnedSessionIds}
        searchQuery={sessionSearch}
        onSearchChange={handleSearchChange}
        className="hidden md:flex"
      />

      {/* Right: main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* must_change_password banner */}
        {mustChangePassword && (
          <div className="bg-amber-50 border-b border-amber-200 px-4 py-3 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-600" />
              <span className="text-amber-800 text-sm font-medium">
                You are required to change your password.
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="border-amber-300 text-amber-700 hover:bg-amber-100 hover:text-amber-800"
              onClick={() => navigate("/settings?action=change-password")}
            >
              Change Password
            </Button>
          </div>
        )}
        {children}
      </main>
    </div>
  );
}
