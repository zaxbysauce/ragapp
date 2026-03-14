import { BrowserRouter, Routes, Route, useNavigate, useSearchParams } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { PageShell } from "@/components/layout/PageShell";
import ChatPage from "@/pages/ChatPage";
import ChatPageRedesigned from "@/pages/ChatPageRedesigned";
import DocumentsPage from "@/pages/DocumentsPage";
import MemoryPage from "@/pages/MemoryPage";
import VaultsPage from "@/pages/VaultsPage";
import SettingsPage from "@/pages/SettingsPage";
import LoginPage from "@/pages/LoginPage";
import { useHealthCheck } from "@/hooks/useHealthCheck";
import { useState } from "react";

type PageId = "chat" | "documents" | "memory" | "vaults" | "settings";

const pages: Record<PageId, React.ComponentType> = {
  chat: ChatPage,
  documents: DocumentsPage,
  memory: MemoryPage,
  vaults: VaultsPage,
  settings: SettingsPage,
};

function MainApp() {
  const [searchParams] = useSearchParams();
  const rawPage = searchParams.get("page") as PageId | null;
  const initialPage: PageId = rawPage && rawPage in pages ? rawPage : "chat";
  const [activePage, setActivePage] = useState<PageId>(initialPage);
  const health = useHealthCheck({ pollInterval: 30000 });

  const CurrentPage = pages[activePage];

  return (
    <PageShell
      activeItem={activePage}
      onItemSelect={(id) => setActivePage(id as PageId)}
      healthStatus={health}
    >
      <CurrentPage />
    </PageShell>
  );
}

/** Shell for the redesigned chat page — routes nav clicks back to the main app. */
function RedesignShell() {
  const navigate = useNavigate();
  const health = useHealthCheck({ pollInterval: 30000 });

  return (
    <PageShell
      activeItem="chatNew"
      onItemSelect={(id) => {
        if (id === "chatNew") return;
        navigate(`/?page=${id}`);
      }}
      healthStatus={health}
      noPadding
    >
      <ChatPageRedesigned />
    </PageShell>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/chat/redesign"
            element={
              <ProtectedRoute>
                <RedesignShell />
              </ProtectedRoute>
            }
          />
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <MainApp />
              </ProtectedRoute>
            }
          />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
