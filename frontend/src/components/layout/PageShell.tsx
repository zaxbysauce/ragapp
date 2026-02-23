import { ReactNode } from "react";
import { Navigation } from "./Navigation";
import type { HealthStatus } from "@/types/health";
import type { NavItemId } from "./navigationTypes";

interface PageShellProps {
  children: ReactNode;
  activeItem: NavItemId;
  onItemSelect: (id: NavItemId) => void;
  healthStatus: HealthStatus;
}

export function PageShell({ children, activeItem, onItemSelect, healthStatus }: PageShellProps) {
  return (
    <div className="flex min-h-screen">
      {/* Navigation - Responsive (Desktop Rail / Mobile Bottom Nav) */}
      <Navigation activeItem={activeItem} onItemSelect={onItemSelect} healthStatus={healthStatus} />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-h-screen overflow-hidden">
        <div className="flex-1 p-6 lg:p-8 overflow-auto pb-20 md:pb-6">
          {children}
        </div>
      </main>
    </div>
  );
}
