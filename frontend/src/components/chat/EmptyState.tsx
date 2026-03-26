import { motion } from "framer-motion";
import { EmptyTranscript } from "./EmptyTranscript";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  vaultName: string;
  hasDocuments: boolean;
  onPromptClick: (prompt: string) => void;
  onUploadClick?: () => void;
  className?: string;
}

export function EmptyState({
  vaultName,
  hasDocuments,
  onPromptClick,
  onUploadClick,
  className,
}: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center", className)}>
      {/* Vault Name */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-muted-foreground text-sm font-medium mb-2"
      >
        {vaultName}
      </motion.p>

      {/* Empty Transcript Content */}
      <EmptyTranscript
        hasDocuments={hasDocuments}
        onPromptClick={onPromptClick}
        onUploadClick={onUploadClick}
      />
    </div>
  );
}
