"use client";

import { useState } from "react";
import { FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { Source } from "@/lib/api";

interface SourceCitationsProps {
  sources: Source[];
}

interface InlineCitationProps {
  sourceIndex: number;
  onClick?: () => void;
}

interface SourceDetailDialogProps {
  source: Source | null;
  open: boolean;
  onClose: () => void;
}

/**
 * Get color class for score badge based on score value.
 * - Green: score > 0.8
 * - Yellow: score 0.5 - 0.8
 * - Red: score < 0.5
 */
function getScoreColorClass(score: number | undefined): string {
  if (score === undefined || score === null) {
    return "text-gray-500";
  }
  if (score > 0.8) {
    return "text-green-600";
  }
  if (score >= 0.5) {
    return "text-yellow-600";
  }
  return "text-red-600";
}

/**
 * Format score for display.
 */
function formatScore(score: number | undefined): string {
  if (score === undefined || score === null) {
    return "N/A";
  }
  return score.toFixed(2);
}

/**
 * Inline citation badge component.
 * Renders a superscript-style badge like [1] [2] that can be clicked to open source detail.
 */
export function InlineCitation({ sourceIndex, onClick }: InlineCitationProps) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center justify-center px-1 py-0 mx-0.5 text-xs font-medium text-primary bg-primary/10 rounded hover:bg-primary/20 transition-colors align-super"
      aria-label={`View source ${sourceIndex + 1}`}
    >
      [{sourceIndex + 1}]
    </button>
  );
}

/**
 * Source detail dialog component.
 * Shows full source information when a citation is clicked.
 */
export function SourceDetailDialog({
  source,
  open,
  onClose,
}: SourceDetailDialogProps) {
  if (!source) {
    return null;
  }

  const scoreColorClass = getScoreColorClass(source.score);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            {source.filename}
          </DialogTitle>
          <DialogDescription>
            Source details and relevance score
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 mt-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Score:</span>
              <Badge
                variant="outline"
                className={`font-mono ${scoreColorClass}`}
              >
                {formatScore(source.score)}
              </Badge>
            </div>
            {source.score_type && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Type:</span>
                <Badge variant="secondary" className="text-xs capitalize">
                  {source.score_type}
                </Badge>
              </div>
            )}
          </div>

          {source.snippet && (
            <div className="space-y-2">
              <span className="text-sm font-medium text-foreground">
                Excerpt:
              </span>
              <div className="rounded-md border bg-muted p-4">
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {source.snippet}
                </p>
              </div>
            </div>
          )}

          <div className="text-xs text-muted-foreground">
            Source ID: {source.id}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Source chip component for the evidence strip.
 * Shows filename and colored score badge.
 */
function SourceChip({
  source,
  onClick,
}: {
  source: Source;
  onClick: () => void;
}) {
  const scoreColorClass = getScoreColorClass(source.score);

  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border bg-background hover:bg-accent transition-colors"
      aria-label={`View details for ${source.filename}`}
    >
      <span className="text-xs font-medium text-foreground truncate max-w-[120px]">
        {source.filename}
      </span>
      <Badge
        variant="outline"
        className={`text-xs font-mono px-1.5 py-0 h-5 ${scoreColorClass}`}
      >
        {formatScore(source.score)}
      </Badge>
    </button>
  );
}

/**
 * Source citations component.
 * Renders an evidence strip showing top 3 source chips with filename + score badge.
 * Score colors: green >0.8, yellow 0.5-0.8, red <0.5.
 * Clicking a chip opens source detail dialog.
 */
export function SourceCitations({ sources }: SourceCitationsProps) {
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  // Take top 3 sources for the evidence strip
  const topSources = sources.slice(0, 3);

  const handleSourceClick = (source: Source) => {
    setSelectedSource(source);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedSource(null);
  };

  return (
    <>
      <div className="flex flex-wrap items-center gap-2 mt-3 pt-3 border-t border-border">
        <span className="text-xs text-muted-foreground">Sources:</span>
        {topSources.map((source) => (
          <SourceChip
            key={source.id}
            source={source}
            onClick={() => handleSourceClick(source)}
          />
        ))}
        {sources.length > 3 && (
          <span className="text-xs text-muted-foreground">
            +{sources.length - 3} more
          </span>
        )}
      </div>

      <SourceDetailDialog
        source={selectedSource}
        open={dialogOpen}
        onClose={handleCloseDialog}
      />
    </>
  );
}
