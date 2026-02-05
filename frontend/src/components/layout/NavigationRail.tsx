import { MessageSquare, FileText, Brain, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

export type NavItem = {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
};

const navItems: NavItem[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "memory", label: "Memory", icon: Brain },
  { id: "settings", label: "Settings", icon: Settings },
];

interface HealthStatus {
  backend: boolean;
  embeddings: boolean;
  chat: boolean;
  loading: boolean;
  lastChecked: Date | null;
}

interface NavigationRailProps {
  activeItem: "chat" | "documents" | "memory" | "settings";
  onItemSelect: (id: string) => void;
  healthStatus: HealthStatus;
}

function StatusIndicator({ isUp, label, loading }: { isUp: boolean; label: string; loading?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className={cn(
          "w-2 h-2 rounded-full",
          loading ? "bg-yellow-500 animate-pulse" : isUp ? "bg-green-500" : "bg-red-500"
        )}
      />
      <span className="text-[9px] text-muted-foreground truncate">
        {loading ? "Checking" : label}
      </span>
    </div>
  );
}

export function NavigationRail({ activeItem, onItemSelect, healthStatus }: NavigationRailProps) {
  return (
    <nav className="w-20 min-h-screen bg-card/80 backdrop-blur-sm border-r border-border flex flex-col items-center py-6 gap-2">
      {/* App Logo */}
      <div className="mb-8 p-3 rounded-xl bg-primary/10">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <span className="text-primary-foreground font-bold text-sm">KV</span>
        </div>
      </div>

      {/* Navigation Items */}
      <div className="flex flex-col gap-1 w-full px-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeItem === item.id;

          return (
            <button
              key={item.id}
              onClick={() => onItemSelect(item.id)}
              className={cn(
                "group relative flex flex-col items-center gap-1 p-3 rounded-xl transition-all duration-200 ease-out",
                "hover:bg-secondary focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                isActive && "bg-primary/10"
              )}
              aria-label={item.label}
              aria-current={isActive ? "page" : undefined}
            >
              {/* Icon Container */}
              <div
                className={cn(
                  "relative p-2 rounded-lg transition-all duration-200",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground group-hover:text-foreground"
                )}
              >
                <Icon className="w-5 h-5" />
                
                {/* Active Indicator */}
                {isActive && (
                  <span className="absolute -right-1 top-1/2 -translate-y-1/2 w-1 h-4 bg-primary rounded-full" />
                )}
              </div>

              {/* Label */}
              <span
                className={cn(
                  "text-[10px] font-medium transition-colors duration-200",
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground group-hover:text-foreground"
                )}
              >
                {item.label}
              </span>

              {/* Tooltip for larger screens */}
              <span className="sr-only">{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* Bottom Spacer */}
      <div className="mt-auto" />

      {/* Health Status Footer */}
      <div className="w-full px-2 pb-4">
        <div className="flex flex-col gap-1.5 p-2 rounded-lg bg-muted/50">
          <StatusIndicator isUp={healthStatus.backend} label="API" loading={healthStatus.loading} />
          <StatusIndicator isUp={healthStatus.embeddings} label="Embeddings" loading={healthStatus.loading} />
          <StatusIndicator isUp={healthStatus.chat} label="Chat" loading={healthStatus.loading} />
        </div>
      </div>
    </nav>
  );
}
