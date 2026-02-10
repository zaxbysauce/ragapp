import { ReactNode } from "react";
import { NavigationRail } from "./NavigationRail";
import type { HealthStatus } from "@/types/health";

interface PageShellProps {
  children: ReactNode;
  activeItem: "chat" | "documents" | "memory" | "vaults" | "settings";
  onItemSelect: (id: string) => void;
  healthStatus: HealthStatus;
}

export function PageShell({ children, activeItem, onItemSelect, healthStatus }: PageShellProps) {
  return (
    <div className="flex min-h-screen">
      {/* Navigation Rail */}
      <NavigationRail activeItem={activeItem} onItemSelect={onItemSelect} healthStatus={healthStatus} />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-h-screen overflow-hidden">
        <div className="flex-1 p-6 lg:p-8 overflow-auto">
          {children}
        </div>
      </main>
    </div>
  );
}
