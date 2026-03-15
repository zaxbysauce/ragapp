import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, X, Highlighter, ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

interface PreviewTabProps {
  sources?: Source[];
  focusedSourceId?: string | null;
  onFocusSource?: (sourceId: string | null) => void;
}

// Mock document content - in real implementation this would come from API
function generateMockContent(filename: string): string {
  return `# ${filename}

## Introduction

This is a sample document that demonstrates the preview functionality. In a real implementation, this content would be fetched from the backend API based on the document ID.

## Key Points

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Detailed Analysis

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Conclusion

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.

---

Additional content would appear here based on the actual document content retrieved from the server. The highlighting feature would search through this content and highlight matching terms or the specific snippet that was referenced in the RAG response.
`;
}

export function PreviewTab({
  sources,
  focusedSourceId,
  onFocusSource,
}: PreviewTabProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const [zoom, setZoom] = useState(100);

  // Get focused source or first source
  const activeSource = focusedSourceId
    ? sources?.find((s) => s.id === focusedSourceId)
    : sources?.[0];

  // Generate mock content for the active source
  const content = activeSource ? generateMockContent(activeSource.filename) : "";

  // Find all matches of the search query (usually the snippet)
  const matches = searchQuery
    ? content
        .toLowerCase()
        .split(searchQuery.toLowerCase())
        .map((_, i, arr) => (i < arr.length - 1 ? i : -1))
        .filter((i) => i !== -1)
    : [];

  // Update search query when source changes (use snippet as search)
  useEffect(() => {
    if (activeSource?.snippet) {
      // Use first few words of snippet as search query
      const words = activeSource.snippet.split(/\s+/).slice(0, 5).join(" ");
      setSearchQuery(words);
      setCurrentMatchIndex(0);
    } else {
      setSearchQuery("");
    }
  }, [activeSource]);

  const navigateMatch = (direction: "prev" | "next") => {
    if (matches.length === 0) return;
    if (direction === "next") {
      setCurrentMatchIndex((prev) => (prev + 1) % matches.length);
    } else {
      setCurrentMatchIndex((prev) => (prev - 1 + matches.length) % matches.length);
    }
  };

  // Render content with highlights
  const renderHighlightedContent = () => {
    if (!searchQuery || matches.length === 0) {
      return <pre className="whitespace-pre-wrap font-mono text-sm">{content}</pre>;
    }

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    matches.forEach((matchIndex, i) => {
      const matchStart = content
        .toLowerCase()
        .indexOf(searchQuery.toLowerCase(), lastIndex);
      const matchEnd = matchStart + searchQuery.length;

      // Add text before match
      if (matchStart > lastIndex) {
        parts.push(
          <span key={`text-${i}`}>
            {content.slice(lastIndex, matchStart)}
          </span>
        );
      }

      // Add highlighted match
      const isCurrentMatch = i === currentMatchIndex;
      parts.push(
        <mark
          key={`match-${i}`}
          className={cn(
            "rounded px-0.5 transition-colors",
            isCurrentMatch
              ? "bg-yellow-300 text-yellow-900 dark:bg-yellow-600 dark:text-yellow-100"
              : "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200"
          )}
        >
          {content.slice(matchStart, matchEnd)}
        </mark>
      );

      lastIndex = matchEnd;
    });

    // Add remaining text
    if (lastIndex < content.length) {
      parts.push(
        <span key="text-end">{content.slice(lastIndex)}</span>
      );
    }

    return (
      <pre className="whitespace-pre-wrap font-mono text-sm">{parts}</pre>
    );
  };

  if (!sources || sources.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-6">
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-3">
          <FileText className="h-6 w-6 text-muted-foreground" />
        </div>
        <p className="text-sm text-muted-foreground">No preview available</p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          Select a document to preview
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with source selector */}
      <div className="border-b border-border p-3 space-y-3">
        {/* Source tabs */}
        <div className="flex gap-1 overflow-x-auto pb-1">
          {sources.map((source) => (
            <button
              key={source.id}
              onClick={() => onFocusSource?.(source.id)}
              className={cn(
                "flex items-center gap-1.5 px-2 py-1 rounded-md text-xs whitespace-nowrap transition-colors",
                focusedSourceId === source.id
                  ? "bg-primary/10 text-primary"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              )}
            >
              <FileText className="h-3 w-3" />
              <span className="max-w-[100px] truncate">{source.filename}</span>
            </button>
          ))}
        </div>

        {/* Toolbar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setZoom((z) => Math.max(50, z - 10))}
              disabled={zoom <= 50}
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </Button>
            <span className="text-xs text-muted-foreground w-12 text-center">
              {zoom}%
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setZoom((z) => Math.min(200, z + 10))}
              disabled={zoom >= 200}
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </Button>
          </div>

          {matches.length > 0 && (
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => navigateMatch("prev")}
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </Button>
              <span className="text-xs text-muted-foreground">
                {currentMatchIndex + 1} / {matches.length}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => navigateMatch("next")}
              >
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Document content */}
      <ScrollArea className="flex-1">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeSource?.id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-4"
            style={{ fontSize: `${zoom}%` }}
          >
            {renderHighlightedContent()}
          </motion.div>
        </AnimatePresence>
      </ScrollArea>
    </div>
  );
}
