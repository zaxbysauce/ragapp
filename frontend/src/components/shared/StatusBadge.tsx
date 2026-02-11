import { Badge } from "@/components/ui/badge";
import { CheckCircle, Loader2, Clock, AlertCircle } from "lucide-react";

interface StatusBadgeProps {
  status?: string;
}

/** Renders a color-coded badge for document processing status. */
export function StatusBadge({ status }: StatusBadgeProps) {
  switch (status) {
    case "processed":
      return (
        <Badge variant="default" className="bg-green-500">
          <CheckCircle className="w-3 h-3 mr-1" />
          Processed
        </Badge>
      );
    case "processing":
      return (
        <Badge variant="secondary">
          <Loader2 className="w-3 h-3 mr-1 animate-spin" />
          Processing
        </Badge>
      );
    case "pending":
      return (
        <Badge variant="outline">
          <Clock className="w-3 h-3 mr-1" />
          Pending
        </Badge>
      );
    case "error":
      return (
        <Badge variant="destructive">
          <AlertCircle className="w-3 h-3 mr-1" />
          Error
        </Badge>
      );
    default:
      return <Badge variant="outline">Unknown</Badge>;
  }
}
