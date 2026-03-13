import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import { Copy, Check } from "lucide-react";
import type { Source } from "@/lib/api";

interface MessageContentProps {
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}

const escapeHtml = (unsafe: string): string => {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
};

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-border">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-muted/80 border-b border-border">
        <span className="text-xs font-mono text-muted-foreground">
          {language || "code"}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-green-500" />
              <span className="text-green-500">Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <pre className="overflow-x-auto bg-[#1a1a1a] dark:bg-[#111] p-4 text-sm">
        <code className="font-mono text-[#e4e4e4] leading-relaxed">{children}</code>
      </pre>
    </div>
  );
}

export function MessageContent({ content, sources, isStreaming }: MessageContentProps) {
  return (
    <div>
      <div className="prose prose-sm dark:prose-invert max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSanitize]}
          components={{
            code({ className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              const isBlock = !!match || (typeof children === "string" && children.includes("\n"));

              if (isBlock) {
                return (
                  <CodeBlock language={match ? match[1] : ""}>
                    {String(children).replace(/\n$/, "")}
                  </CodeBlock>
                );
              }

              // Inline code
              return (
                <code
                  className="bg-muted rounded px-1.5 py-0.5 text-sm font-mono text-foreground"
                  {...props}
                >
                  {children}
                </code>
              );
            },
          }}
        >
          {content}
        </ReactMarkdown>
        {isStreaming && (
          <span className="inline-block w-2 h-4 ml-1 bg-foreground/70 animate-pulse rounded-sm" />
        )}
      </div>

      {sources && sources.length > 0 && (
        <div className="mt-4 pt-4 border-t border-border">
          <p className="text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">
            Sources
          </p>
          <div className="flex flex-wrap gap-2">
            {sources.map((source, i) => (
              <div
                key={source.id}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-muted/60 border border-border/60 hover:bg-muted transition-colors"
                title={source.snippet || source.filename}
              >
                <span className="w-4 h-4 rounded-full bg-primary/20 text-primary flex items-center justify-center text-[9px] font-bold shrink-0">
                  {i + 1}
                </span>
                <span className="font-medium truncate max-w-[160px]">{source.filename}</span>
                {source.score !== undefined && (
                  <span className="text-muted-foreground/60">
                    {Math.round(source.score * 100)}%
                  </span>
                )}
              </div>
            ))}
          </div>

          {/* Snippet details for expanded view */}
          {sources.some((s) => s.snippet) && (
            <div className="mt-3 space-y-2">
              {sources.map((source, i) =>
                source.snippet ? (
                  <div
                    key={source.id}
                    className="text-xs p-2.5 rounded-lg bg-muted/40 border border-border/40"
                  >
                    <span className="font-semibold text-muted-foreground">
                      [{i + 1}] {source.filename}
                    </span>
                    <p
                      className="mt-1 text-muted-foreground line-clamp-2"
                      dangerouslySetInnerHTML={{ __html: escapeHtml(source.snippet) }}
                    />
                  </div>
                ) : null
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
