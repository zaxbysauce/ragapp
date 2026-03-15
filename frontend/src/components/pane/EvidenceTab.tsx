import { motion } from "framer-motion";
import { FileText, ChevronRight, BookOpen } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

interface EvidenceTabProps {
  sources?: Source[];
  focusedSourceId?: string | null;
  onFocusSource?: (sourceId: string | null) => void;
}

function getRelevanceLabel(score: number, scoreType?: string): string {
  switch (scoreType) {
    case "distance":
      if (score < 0.2) return "Very High";
      if (score < 0.4) return "High";
      if (score < 0.6) return "Medium";
      return "Low";
    case "rerank":
      if (score > 0.8) return "Very High";
      if (score > 0.6) return "High";
      if (score > 0.4) return "Medium";
      return "Low";
    case "rrf":
      if (score > 0.7) return "Very High";
      if (score > 0.5) return "High";
      if (score > 0.3) return "Medium";
      return "Low";
    default:
      if (score > 0.8) return "Very High";
      if (score > 0.6) return "High";
      if (score > 0.4) return "Medium";
      return "Low";
  }
}

function getRelevanceColor(label: string): string {
  switch (label) {
    case "Very High":
      return "bg-green-500/10 text-green-600 border-green-500/20";
    case "High":
      return "bg-blue-500/10 text-blue-600 border-blue-500/20";
    case "Medium":
      return "bg-yellow-500/10 text-yellow-600 border-yellow-500/20";
    default:
      return "bg-muted text-muted-foreground";
  }
}

function getRankIcon(rank: number): string {
  switch (rank) {
    case 1:
      return "🥇";
    case 2:
      return "🥈";
    case 3:
      return "🥉";
    default:
      return `#${rank}`;
  }
}

export function EvidenceTab({
  sources,
  focusedSourceId,
  onFocusSource,
}: EvidenceTabProps) {
  if (!sources || sources.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-6">
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-3">
          <BookOpen className="h-6 w-6 text-muted-foreground" />
        </div>
        <p className="text-sm text-muted-foreground">No evidence available</p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          Ask a question to see relevant sources
        </p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-4 space-y-3">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs text-muted-foreground">
            {sources.length} source{sources.length !== 1 ? "s" : ""} found
          </p>
          <div className="flex gap-2">
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Rank
            </span>
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Relevance
            </span>
          </div>
        </div>

        {sources.map((source, index) => {
          const isFocused = focusedSourceId === source.id;
          const relevance = source.score
            ? getRelevanceLabel(source.score, source.score_type)
            : null;
          const rank = index + 1;

          return (
            <motion.div
              key={source.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className={cn(
                "rounded-lg border transition-all cursor-pointer",
                isFocused
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/50"
              )}
              onClick={() => onFocusSource?.(isFocused ? null : source.id)}
            >
              <div className="p-3">
                <div className="flex items-start gap-2">
                  {/* Rank indicator */}
                  <div className="flex flex-col items-center gap-1 shrink-0 w-8">
                    <span className="text-lg" title={`Rank ${rank}`}>
                      {getRankIcon(rank)}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      #{rank}
                    </span>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <FileText className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                      <span className="text-sm font-medium truncate">
                        {source.filename}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      {relevance && (
                        <Badge
                          variant="outline"
                          className={cn("text-[10px]", getRelevanceColor(relevance))}
                        >
                          {relevance}
                        </Badge>
                      )}
                      {source.score !== undefined && (
                        <span className="text-[10px] text-muted-foreground tabular-nums">
                          {source.score_type === "distance"
                            ? `${source.score.toFixed(3)}`
                            : `${(source.score * 100).toFixed(1)}%`}
                        </span>
                      )}
                    </div>
                  </div>

                  <ChevronRight
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform shrink-0",
                      isFocused && "rotate-90"
                    )}
                  />
                </div>

                {/* Expanded snippet */}
                {isFocused && source.snippet && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="pt-3 mt-3 border-t border-border/50">
                      <p className="text-xs text-muted-foreground whitespace-pre-wrap leading-relaxed">
                        {source.snippet}
                      </p>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </ScrollArea>
  );
}
