import { motion, AnimatePresence } from "framer-motion";
import { X, PanelRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";
import { type RightPaneTab } from "@/stores/chatShellStore";
import { EvidenceTab } from "./EvidenceTab";
import { PreviewTab } from "./PreviewTab";
import { WorkspaceTab } from "./WorkspaceTab";

interface RightPaneProps {
  isOpen: boolean;
  activeTab: RightPaneTab;
  onTabChange: (tab: RightPaneTab) => void;
  onClose: () => void;
  sources?: Source[];
  focusedSourceId?: string | null;
  onFocusSource?: (sourceId: string | null) => void;
  className?: string;
}

export function RightPane({
  isOpen,
  activeTab,
  onTabChange,
  onClose,
  sources,
  focusedSourceId,
  onFocusSource,
  className,
}: RightPaneProps) {
  return (
    <AnimatePresence mode="wait">
      {isOpen && (
        <motion.aside
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 360, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.25, ease: "easeInOut" }}
          className={cn(
            "h-full bg-card border-l border-border flex flex-col overflow-hidden shrink-0",
            className
          )}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
            <div className="flex items-center gap-2">
              <PanelRight className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-semibold">Details</span>
            </div>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Tabs */}
          <Tabs
            value={activeTab}
            onValueChange={(v) => onTabChange(v as RightPaneTab)}
            className="flex-1 flex flex-col min-h-0"
          >
            <TabsList className="w-full justify-start rounded-none border-b border-border bg-transparent px-4 py-2 shrink-0">
              <TabsTrigger
                value="evidence"
                className="text-xs data-[state=active]:bg-primary/10 data-[state=active]:text-primary"
              >
                Evidence
              </TabsTrigger>
              <TabsTrigger
                value="preview"
                className="text-xs data-[state=active]:bg-primary/10 data-[state=active]:text-primary"
              >
                Preview
              </TabsTrigger>
              <TabsTrigger
                value="workspace"
                className="text-xs data-[state=active]:bg-primary/10 data-[state=active]:text-primary"
              >
                Workspace
              </TabsTrigger>
            </TabsList>

            <div className="flex-1 min-h-0 overflow-hidden">
              <TabsContent value="evidence" className="h-full m-0 data-[state=inactive]:hidden">
                <EvidenceTab
                  sources={sources}
                  focusedSourceId={focusedSourceId}
                  onFocusSource={onFocusSource}
                />
              </TabsContent>

              <TabsContent value="preview" className="h-full m-0 data-[state=inactive]:hidden">
                <PreviewTab
                  sources={sources}
                  focusedSourceId={focusedSourceId}
                  onFocusSource={onFocusSource}
                />
              </TabsContent>

              <TabsContent value="workspace" className="h-full m-0 data-[state=inactive]:hidden">
                <WorkspaceTab sources={sources} />
              </TabsContent>
            </div>
          </Tabs>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
