import { FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

interface DocumentPreviewProps {
  source: Source;
}

// Simple HTML escape function to prevent XSS
const escapeHtml = (text: string | undefined): string => {
  if (!text) return "";
  return text.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      case "'":
        return "&#x27;";
      default:
        return char;
    }
  });
};

export function DocumentPreview({ source }: DocumentPreviewProps) {
  // Note: The existing Source type doesn't have a content field, so we display the snippet
  // In a real implementation with editing, you would extend the Source type

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border">
        <FileText className="h-5 w-5 text-muted-foreground" />
        <span className="font-medium truncate">{source.filename}</span>
      </div>

      <div className="flex-1 overflow-auto">
        <div className={cn("prose prose-sm dark:prose-invert max-w-none", "p-4 rounded-lg bg-muted/30")}>
          <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
            {escapeHtml(source.snippet) || "No content available"}
          </pre>
        </div>
      </div>
    </div>
  );
}
