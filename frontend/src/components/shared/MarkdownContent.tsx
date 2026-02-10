import ReactMarkdown from "react-markdown";
import React from "react";

const MARKDOWN_COMPONENTS = {
  code({ className, children, ...props }: any) {
    const isInline = !className;
    return isInline ? (
      <code className="bg-muted px-1 py-0.5 rounded text-sm font-mono" {...props}>
        {children}
      </code>
    ) : (
      <pre className="bg-muted p-3 rounded-lg overflow-x-auto my-2">
        <code className="text-sm font-mono" {...props}>
          {children}
        </code>
      </pre>
    );
  },
  ul({ children }: any) {
    return <ul className="list-disc pl-5 my-2">{children}</ul>;
  },
  ol({ children }: any) {
    return <ol className="list-decimal pl-5 my-2">{children}</ol>;
  },
  li({ children }: any) {
    return <li className="my-0.5">{children}</li>;
  },
  p({ children }: any) {
    return <p className="my-2">{children}</p>;
  },
  h1({ children }: any) {
    return <h1 className="text-xl font-bold my-3">{children}</h1>;
  },
  h2({ children }: any) {
    return <h2 className="text-lg font-bold my-2">{children}</h2>;
  },
  h3({ children }: any) {
    return <h3 className="text-base font-bold my-2">{children}</h3>;
  },
  blockquote({ children }: any) {
    return <blockquote className="border-l-2 border-muted-foreground pl-3 italic my-2">{children}</blockquote>;
  },
};

export const MarkdownContent = React.memo(function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown
        components={MARKDOWN_COMPONENTS}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
