import type { HealthStatus } from "@/types/health";
import type { ComponentType } from "react";

export type NavItemId = "chat" | "documents" | "memory" | "vaults" | "settings";

export interface NavItem {
  id: NavItemId;
  label: string;
  icon: ComponentType<{ className?: string }>;
}

export interface NavigationProps {
  activeItem: NavItemId;
  onItemSelect: (id: NavItemId) => void;
  healthStatus: HealthStatus;
}
