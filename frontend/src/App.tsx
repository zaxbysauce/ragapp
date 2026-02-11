import { useState } from "react";
import { PageShell } from "@/components/layout/PageShell";
import ChatPage from "@/pages/ChatPage";
import DocumentsPage from "@/pages/DocumentsPage";
import MemoryPage from "@/pages/MemoryPage";
import VaultsPage from "@/pages/VaultsPage";
import SettingsPage from "@/pages/SettingsPage";
import { useHealthCheck } from "@/hooks/useHealthCheck";

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

export default App;
