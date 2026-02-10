import { useState, useEffect } from "react";
import { PageShell } from "@/components/layout/PageShell";
import { getHealth } from "@/lib/api";
import ChatPage from "@/pages/ChatPage";
import DocumentsPage from "@/pages/DocumentsPage";
import MemoryPage from "@/pages/MemoryPage";
import VaultsPage from "@/pages/VaultsPage";
import SettingsPage from "@/pages/SettingsPage";
import type { HealthStatus } from "@/types/health";

type PageId = "chat" | "documents" | "memory" | "vaults" | "settings";

const pages: Record<PageId, React.ComponentType> = {
  chat: ChatPage,
  documents: DocumentsPage,
  memory: MemoryPage,
  vaults: VaultsPage,
  settings: SettingsPage,
};

function App() {
  const [activePage, setActivePage] = useState<PageId>("chat");
  const [health, setHealth] = useState<HealthStatus>({
    backend: false,
    embeddings: false,
    chat: false,
    loading: true,
    lastChecked: null,
  });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await getHealth();
        setHealth({
          backend: response.services?.backend ?? response.status === "ok",
          embeddings: response.services?.embeddings ?? false,
          chat: response.services?.chat ?? false,
          loading: false,
          lastChecked: new Date(),
        });
      } catch {
        setHealth({
          backend: false,
          embeddings: false,
          chat: false,
          loading: false,
          lastChecked: new Date(),
        });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

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

export default App;
