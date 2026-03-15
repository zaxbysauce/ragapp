import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutGrid,
  List,
  Table,
  Download,
  Copy,
  Check,
  FileJson,
  FileSpreadsheet,
  FileText,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/api";

interface WorkspaceTabProps {
  sources?: Source[];
}

type OutputFormat = "table" | "list" | "json";

interface StructuredOutput {
  title: string;
  type: "summary" | "comparison" | "facts" | "custom";
  createdAt: string;
  content: unknown;
}

// Mock structured outputs - in real implementation these would come from API
function generateMockOutputs(): StructuredOutput[] {
  return [
    {
      title: "Key Facts Summary",
      type: "facts",
      createdAt: new Date().toISOString(),
      content: {
        facts: [
          { fact: "Revenue increased by 25% in Q4", source: "financial_report.pdf" },
          { fact: "New product line launched", source: "product_update.docx" },
          { fact: "Customer satisfaction at 94%", source: "survey_results.pdf" },
        ],
      },
    },
    {
      title: "Document Comparison",
      type: "comparison",
      createdAt: new Date(Date.now() - 86400000).toISOString(),
      content: {
        comparisons: [
          { aspect: "Approach", doc1: "Agile", doc2: "Waterfall" },
          { aspect: "Timeline", doc1: "3 months", doc2: "6 months" },
          { aspect: "Budget", doc1: "$50k", doc2: "$75k" },
        ],
      },
    },
  ];
}

function TableView({ data }: { data: unknown }) {
  if (!data || typeof data !== "object") return null;

  const entries = Object.entries(data);
  if (entries.length === 0) return null;

  // Check if it's an array of objects
  const firstValue = entries[0]?.[1];
  if (Array.isArray(firstValue) && firstValue.length > 0 && typeof firstValue[0] === "object") {
    const array = firstValue as Record<string, string>[];
    const columns = Object.keys(array[0]);

    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              {columns.map((col) => (
                <th key={col} className="text-left py-2 px-3 font-medium text-muted-foreground">
                  {col.charAt(0).toUpperCase() + col.slice(1)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {array.map((row, i) => (
              <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                {columns.map((col) => (
                  <td key={col} className="py-2 px-3">
                    {row[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return null;
}

function ListView({ data }: { data: unknown }) {
  if (!data || typeof data !== "object") return null;

  const entries = Object.entries(data);

  return (
    <div className="space-y-4">
      {entries.map(([key, value]) => (
        <div key={key} className="border border-border rounded-lg p-3">
          <h4 className="text-sm font-medium mb-2">
            {key.charAt(0).toUpperCase() + key.slice(1)}
          </h4>
          {Array.isArray(value) ? (
            <ul className="space-y-1">
              {value.map((item, i) => (
                <li key={i} className="text-sm text-muted-foreground">
                  {typeof item === "object" ? JSON.stringify(item) : String(item)}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">{String(value)}</p>
          )}
        </div>
      ))}
    </div>
  );
}

function JsonView({ data }: { data: unknown }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        className="absolute top-2 right-2 gap-1"
        onClick={handleCopy}
      >
        {copied ? (
          <>
            <Check className="h-3.5 w-3.5 text-green-500" />
            <span className="text-xs text-green-500">Copied</span>
          </>
        ) : (
          <>
            <Copy className="h-3.5 w-3.5" />
            <span className="text-xs">Copy</span>
          </>
        )}
      </Button>
      <pre className="text-xs font-mono bg-muted p-4 rounded-lg overflow-auto">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

function OutputCard({
  output,
  onSelect,
  isSelected,
}: {
  output: StructuredOutput;
  onSelect: () => void;
  isSelected: boolean;
}) {
  const getIcon = () => {
    switch (output.type) {
      case "summary":
        return <FileText className="h-4 w-4" />;
      case "comparison":
        return <FileSpreadsheet className="h-4 w-4" />;
      case "facts":
        return <List className="h-4 w-4" />;
      default:
        return <FileJson className="h-4 w-4" />;
    }
  };

  const getTypeColor = () => {
    switch (output.type) {
      case "summary":
        return "bg-blue-500/10 text-blue-600";
      case "comparison":
        return "bg-purple-500/10 text-purple-600";
      case "facts":
        return "bg-green-500/10 text-green-600";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left p-3 rounded-lg border transition-all",
        isSelected
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/30 hover:bg-muted/30"
      )}
    >
      <div className="flex items-start gap-3">
        <div className={cn("p-1.5 rounded", getTypeColor())}>{getIcon()}</div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{output.title}</p>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="secondary" className="text-[10px]">
              {output.type}
            </Badge>
            <span className="text-[10px] text-muted-foreground">
              {new Date(output.createdAt).toLocaleDateString()}
            </span>
          </div>
        </div>
      </div>
    </button>
  );
}

export function WorkspaceTab({ sources: _sources }: WorkspaceTabProps) {
  const [selectedOutput, setSelectedOutput] = useState<StructuredOutput | null>(null);
  const [viewMode, setViewMode] = useState<OutputFormat>("table");
  const outputs = generateMockOutputs();

  const activeOutput = selectedOutput || outputs[0];

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-hidden flex">
        {/* Output list */}
        <div className="w-1/2 border-r border-border overflow-y-auto p-3 space-y-2">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">
            Structured Outputs
          </p>
          {outputs.map((output) => (
            <OutputCard
              key={output.title}
              output={output}
              onSelect={() => setSelectedOutput(output)}
              isSelected={activeOutput?.title === output.title}
            />
          ))}

          {outputs.length === 0 && (
            <div className="text-center py-8">
              <LayoutGrid className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">No outputs yet</p>
              <p className="text-xs text-muted-foreground/70">
                Use /summarize or /compare to generate
              </p>
            </div>
          )}
        </div>

        {/* Output content */}
        <div className="w-1/2 flex flex-col">
          {activeOutput ? (
            <>
              <div className="border-b border-border p-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-sm">{activeOutput.title}</h3>
                  <div className="flex items-center gap-1">
                    <Button variant="ghost" size="icon" className="h-7 w-7">
                      <Download className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
                <div className="flex items-center gap-1 mt-2">
                  <button
                    onClick={() => setViewMode("table")}
                    className={cn(
                      "p-1.5 rounded transition-colors",
                      viewMode === "table"
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                    title="Table view"
                  >
                    <Table className="h-3.5 w-3.5" />
                  </button>
                  <button
                    onClick={() => setViewMode("list")}
                    className={cn(
                      "p-1.5 rounded transition-colors",
                      viewMode === "list"
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                    title="List view"
                  >
                    <List className="h-3.5 w-3.5" />
                  </button>
                  <button
                    onClick={() => setViewMode("json")}
                    className={cn(
                      "p-1.5 rounded transition-colors",
                      viewMode === "json"
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                    title="JSON view"
                  >
                    <FileJson className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              <ScrollArea className="flex-1 p-3">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={viewMode}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    {viewMode === "table" && (
                      <TableView data={activeOutput.content} />
                    )}
                    {viewMode === "list" && (
                      <ListView data={activeOutput.content} />
                    )}
                    {viewMode === "json" && (
                      <JsonView data={activeOutput.content} />
                    )}
                  </motion.div>
                </AnimatePresence>
              </ScrollArea>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-sm text-muted-foreground">Select an output</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
