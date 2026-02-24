import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { ConnectionTestResult } from "@/lib/api";

interface ConnectionSettingsProps {
  onTestConnections: () => void;
  isTesting: boolean;
  connectionStatus: ConnectionTestResult | null;
}

export function ConnectionSettings({
  onTestConnections,
  isTesting,
  connectionStatus,
}: ConnectionSettingsProps) {
  return (
    <div className="flex flex-col items-end gap-1">
      <Button size="sm" variant="outline" onClick={onTestConnections} disabled={isTesting}>
        {isTesting ? (
          <Loader2 className="w-3 h-3 animate-spin" />
        ) : (
          "Test Connections"
        )}
      </Button>
      {connectionStatus && (
        <div className="flex gap-2">
          {Object.entries(connectionStatus).map(([service, info]) => (
            <Badge key={service} variant={info.ok ? "outline" : "destructive"} className="text-xs">
              {service}: {info.ok ? "OK" : "Fail"}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
