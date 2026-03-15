import { motion } from "framer-motion";
import {
  Sparkles,
  BookOpen,
  Search,
  Layers,
  HelpCircle,
  Upload,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SuggestedPrompt {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  prompt: string;
}

const SUGGESTED_PROMPTS: SuggestedPrompt[] = [
  {
    icon: BookOpen,
    label: "Summarize documents",
    prompt: "Give me a concise summary of the key topics covered in my documents.",
  },
  {
    icon: Search,
    label: "Find key facts",
    prompt: "What are the most important facts and figures mentioned in my documents?",
  },
  {
    icon: Layers,
    label: "Compare topics",
    prompt: "Are there any conflicting or complementary ideas across my documents? Compare them.",
  },
  {
    icon: HelpCircle,
    label: "Answer a question",
    prompt: "Based on my documents, what can you tell me about ",
  },
];

interface EmptyTranscriptProps {
  hasDocuments: boolean;
  onPromptClick: (prompt: string) => void;
  onUploadClick?: () => void;
  className?: string;
}

export function EmptyTranscript({
  hasDocuments,
  onPromptClick,
  onUploadClick,
  className,
}: EmptyTranscriptProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center min-h-[60vh] px-4 py-12 text-center",
        className
      )}
    >
      {/* Icon */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-4"
      >
        {hasDocuments ? (
          <Sparkles className="h-7 w-7 text-primary" />
        ) : (
          <FileText className="h-7 w-7 text-primary" />
        )}
      </motion.div>

      {/* Title */}
      <motion.h2
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="text-2xl font-semibold tracking-tight mb-1"
      >
        {hasDocuments ? "How can I help you today?" : "No documents yet"}
      </motion.h2>

      {/* Subtitle */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-sm text-muted-foreground mb-8 max-w-sm"
      >
        {hasDocuments
          ? "Ask anything about your documents, or pick a suggestion below."
          : "Upload documents to your vault to start chatting with your knowledge base."}
      </motion.p>

      {/* Content based on state */}
      {hasDocuments ? (
        /* Suggested prompts grid */
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg"
        >
          {SUGGESTED_PROMPTS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.label}
                onClick={() => onPromptClick(item.prompt)}
                className={cn(
                  "flex items-start gap-3 p-4 rounded-xl text-left",
                  "border border-border bg-card hover:bg-muted/50",
                  "transition-colors duration-150 group"
                )}
              >
                <div className="mt-0.5 p-1.5 rounded-lg bg-primary/10 text-primary shrink-0">
                  <Icon className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-sm font-medium group-hover:text-primary transition-colors">
                    {item.label}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                    {item.prompt}
                  </p>
                </div>
              </button>
            );
          })}
        </motion.div>
      ) : (
        /* Upload CTA */
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex flex-col items-center gap-4"
        >
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center">
                <Upload className="h-4 w-4" />
              </div>
              <span>PDF, TXT, DOCX</span>
            </div>
            <div className="h-px w-12 bg-border" />
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center">
                <FileText className="h-4 w-4" />
              </div>
              <span>Auto-indexed</span>
            </div>
          </div>

          <Button
            onClick={onUploadClick}
            size="lg"
            className="gap-2 mt-2"
          >
            <Upload className="h-4 w-4" />
            Upload Documents
          </Button>
        </motion.div>
      )}
    </div>
  );
}
