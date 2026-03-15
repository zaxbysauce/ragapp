import { BrowserRouter, Routes, Route, useNavigate, useSearchParams, Navigate } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { RoleGuard } from "@/components/RoleGuard";
import { PageShell } from "@/components/layout/PageShell";
import { useAuthStore } from "@/stores/authStore";
import ChatShell from "@/pages/ChatShell";
import ChatPage from "@/pages/ChatPage";
import ChatPageRedesigned from "@/pages/ChatPageRedesigned";
import DocumentsPage from "@/pages/DocumentsPage";
import MemoryPage from "@/pages/MemoryPage";
import VaultsPage from "@/pages/VaultsPage";
import SettingsPage from "@/pages/SettingsPage";
import LoginPage from "@/pages/LoginPage";
import RegisterPage from "@/pages/RegisterPage";
import AdminUsersPage from "@/pages/AdminUsersPage";
import OrgsPage from "@/pages/OrgsPage";
import ProfilePage from "@/pages/ProfilePage";
import { useHealthCheck } from "@/hooks/useHealthCheck";
import { useState, useEffect } from "react";

type PageId = "chat" | "documents" | "memory" | "vaults" | "settings" | "admin" | "orgs" | "profile";

const pages: Record<PageId, React.ComponentType> = {
  chat: ChatPage,
  documents: DocumentsPage,
  memory: MemoryPage,
  vaults: VaultsPage,
  settings: SettingsPage,
  admin: AdminUsersPage,
  orgs: OrgsPage,
  profile: ProfilePage,
};

/** Home Surface - Main application pages */
function HomeSurface() {
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

/** Vault Surface - Vault management and member administration */
function VaultSurface() {
  const navigate = useNavigate();
  const health = useHealthCheck({ pollInterval: 30000 });

  return (
    <PageShell
      activeItem="vaults"
      onItemSelect={(id) => {
        if (id === "vaults") return;
        navigate(`/?page=${id}`);
      }}
      healthStatus={health}
    >
      <VaultsPage />
    </PageShell>
  );
}

/** Chat Surface - Dedicated chat interface */
function ChatSurface() {
  const health = useHealthCheck({ pollInterval: 30000 });

  return (
    <PageShell
      activeItem="chat"
      onItemSelect={(id) => {
        if (id !== "chat") {
          window.location.href = `/?page=${id}`;
        }
      }}
      healthStatus={health}
      noPadding
    >
      <ChatShell />
    </PageShell>
  );
}

/** Admin Surface - User and system administration */
function AdminSurface() {
  const [searchParams] = useSearchParams();
  const rawPage = searchParams.get("page") as PageId | null;
  const initialPage: PageId = rawPage && rawPage in pages ? rawPage : "admin";
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
  // Initialize auth store on app load
  useEffect(() => {
    useAuthStore.getState().checkAuth();
  }, []);

  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Auth Routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          
          {/* Four-Surface Navigation */}
          
          {/* 1. Home Surface - Main app with documents, memory, settings */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <HomeSurface />
              </ProtectedRoute>
            }
          />
          
          {/* 2. Vault Surface - Vault management */}
          <Route
            path="/vaults"
            element={
              <ProtectedRoute>
                <VaultSurface />
              </ProtectedRoute>
            }
          />
          
          {/* 3. Chat Surface - Dedicated chat interface */}
          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <ChatSurface />
              </ProtectedRoute>
            }
          />
          <Route
            path="/chat/:sessionId"
            element={
              <ProtectedRoute>
                <ChatSurface />
              </ProtectedRoute>
            }
          />
          
          {/* 4. Admin Surface - User/org management (admin only) */}
          <Route
            path="/admin"
            element={
              <ProtectedRoute>
                <RoleGuard allowedRoles={["admin", "superadmin"]}>
                  <AdminSurface />
                </RoleGuard>
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/*"
            element={
              <ProtectedRoute>
                <RoleGuard allowedRoles={["admin", "superadmin"]}>
                  <AdminSurface />
                </RoleGuard>
              </ProtectedRoute>
            }
          />
          
          {/* Legacy routes for backward compatibility */}
          <Route
            path="/chat/redesign"
            element={
              <ProtectedRoute>
                <RedesignShell />
              </ProtectedRoute>
            }
          />
          
          {/* Profile page - accessible to all authenticated users */}
          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <PageShell
                  activeItem="profile"
                  onItemSelect={(id) => {
                    window.location.href = `/?page=${id}`;
                  }}
                  healthStatus={{ backend: true, embeddings: true, chat: true, loading: false, lastChecked: null }}
                >
                  <ProfilePage />
                </PageShell>
              </ProtectedRoute>
            }
          />
          
          {/* Catch-all redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
