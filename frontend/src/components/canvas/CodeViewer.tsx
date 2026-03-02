import { useState } from "react";
import { Code2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { Source } from "@/lib/api";

interface CodeViewerProps {
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

export function CodeViewer({ source }: CodeViewerProps) {
  const [copied, setCopied] = useState(false);

  // Note: The existing Source type doesn't have a content field, so we simulate live editing
  // with the snippet instead. In a real implementation, you would extend the Source type
  // to include content when editing is enabled.

  const handleCopy = async () => {
    await navigator.clipboard.writeText(source.snippet || "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const language = source.filename?.split('.').pop() || "text";

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Code2 className="h-5 w-5 text-muted-foreground" />
          <span className="font-medium truncate">{source.filename}</span>
          <span className="text-xs px-2 py-0.5 rounded bg-accent text-accent-foreground">
            {language}
          </span>
        </div>
        <Button variant="ghost" size="sm" onClick={handleCopy}>
          {copied ? "Copied!" : "Copy"}
        </Button>
      </div>

      <div className="flex-1 overflow-auto">
        <pre className={cn(
          "p-4 rounded-lg bg-muted/30 overflow-x-auto",
          "text-sm font-mono leading-relaxed"
        )}>
          <code>{escapeHtml(source.snippet) || "No code available"}</code>
        </pre>
      </div>
    </div>
  );
}
