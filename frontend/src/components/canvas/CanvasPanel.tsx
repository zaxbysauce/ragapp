import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Code2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { DocumentPreview } from "./DocumentPreview";
import { CodeViewer } from "./CodeViewer";
import { ResizableHandle } from "./ResizableHandle";
import { Button } from "@/components/ui/button";
import { useChatStore } from "@/stores/useChatStore";

export type CanvasView = "document" | "code" | null;

interface CanvasPanelProps {
  canvas: {
    view: CanvasView;
    isCollapsed: boolean;
    width: number;
  };
  onToggleCollapse: () => void;
  onSetView: (view: CanvasView) => void;
  onSetWidth: (width: number) => void;
}

export function CanvasPanel({ canvas, onToggleCollapse, onSetView, onSetWidth }: CanvasPanelProps) {
  const { messages } = useChatStore();
  const [activeSourceIndex, setActiveSourceIndex] = useState(0);

  // Find last assistant message using reverse and find
  const lastAssistantMessage = [...messages].reverse().find((m) => m.role === "assistant");
  const sources = lastAssistantMessage?.sources || [];

  // Reset active source index when sources change
  useEffect(() => {
    setActiveSourceIndex(0);
  }, [sources]);

  const activeSource = sources[activeSourceIndex];

  const tabs = [
    { id: "document" as const, label: "Document", icon: FileText },
    { id: "code" as const, label: "Code", icon: Code2 },
  ];

  return (
    <AnimatePresence>
      {!canvas.isCollapsed && (
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: canvas.width, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="relative h-full bg-background border-l border-border flex flex-col"
        >
          <ResizableHandle onResize={onSetWidth} />

          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <div className="flex gap-2">
              <div className="flex gap-1">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => onSetView(tab.id)}
                    className={cn(
                      "flex items-center gap-2 px-3 py-1.5 text-sm rounded-md transition-colors",
                      canvas.view === tab.id
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <tab.icon className="h-4 w-4" />
                    {tab.label}
                  </button>
                ))}
              </div>
              {sources.length > 1 && (
                <select
                  value={activeSourceIndex.toString()}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                    setActiveSourceIndex(parseInt(e.target.value))
                  }
                  className="w-[180px] h-8 px-2 text-sm bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-ring"
                >
                  {sources.map((source, index) => (
                    <option key={source.id} value={index.toString()}>
                      {source.filename}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <Button variant="ghost" size="icon" onClick={onToggleCollapse}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex-1 overflow-auto p-4">
            {canvas.view === "document" && activeSource && (
              <DocumentPreview source={activeSource} />
            )}
            {canvas.view === "code" && activeSource && (
              <CodeViewer source={activeSource} />
            )}
            {!activeSource && (
              <div className="h-full flex items-center justify-center text-muted-foreground">
                <p className="text-sm">No document to preview</p>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
