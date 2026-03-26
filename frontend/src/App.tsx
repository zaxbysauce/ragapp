import { BrowserRouter, Routes, Route, useSearchParams, Navigate } from "react-router-dom";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { RoleGuard } from "@/components/RoleGuard";
import { PageShell } from "@/components/layout/PageShell";
import { useAuthStore } from "@/stores/authStore";
import ChatShell from "@/pages/ChatShell";
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

type PageId = "documents" | "memory" | "vaults" | "settings" | "admin" | "orgs" | "profile";

const pages: Record<PageId, React.ComponentType> = {
  documents: DocumentsPage,
  memory: MemoryPage,
  vaults: VaultsPage,
  settings: SettingsPage,
  admin: AdminUsersPage,
  orgs: OrgsPage,
  profile: ProfilePage,
};

/** Home Surface - Main application pages (documents, memory, vaults, settings) */
function HomeSurface() {
  const [searchParams] = useSearchParams();
  const rawPage = searchParams.get("page") as PageId | null;
  const initialPage: PageId = rawPage && rawPage in pages ? rawPage : "documents";
  const [activePage, setActivePage] = useState<PageId>(initialPage);
  const health = useHealthCheck({ pollInterval: 30000 });

  return (
    <PageShell
      activeItem={activePage}
      onItemSelect={(id) => setActivePage(id as PageId)}
      healthStatus={health}
    >
      {(() => {
        const CurrentPage = pages[activePage];
        return <CurrentPage />;
      })()}
    </PageShell>
  );
}

/** Admin Surface - User and organization management (admin only) */
function AdminSurface() {
  const [searchParams] = useSearchParams();
  const rawPage = searchParams.get("page") as PageId | null;
  const initialPage: PageId = rawPage && rawPage in pages ? rawPage : "admin";
  const [activePage, setActivePage] = useState<PageId>(initialPage);
  const health = useHealthCheck({ pollInterval: 30000 });

  return (
    <PageShell
      activeItem={activePage}
      onItemSelect={(id) => setActivePage(id as PageId)}
      healthStatus={health}
    >
      {(() => {
        const CurrentPage = pages[activePage];
        return <CurrentPage />;
      })()}
    </PageShell>
  );
}

function App() {
  useEffect(() => {
    useAuthStore.getState().checkAuth();
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        {/* Auth Routes */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* Chat — canonical chat interface (ChatShell renders its own PageShell) */}
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <ChatShell />
            </ProtectedRoute>
          }
        />
        <Route
          path="/chat/:sessionId"
          element={
            <ProtectedRoute>
              <ChatShell />
            </ProtectedRoute>
          }
        />

        {/* Home Surface - Main app pages */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <HomeSurface />
            </ProtectedRoute>
          }
        />

        {/* Admin Surface */}
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

        {/* Profile page */}
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

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
