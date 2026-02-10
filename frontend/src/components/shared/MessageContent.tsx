import { MarkdownContent } from "./MarkdownContent";
import type { Message } from "@/stores/useChatStore";
import React from "react";

export const MessageContent = React.memo(function MessageContent({ message }: { message: Message }) {
  if (message.role === "assistant") {
    return <MarkdownContent content={message.content} />;
  }
  return <span className="whitespace-pre-wrap">{message.content}</span>;
});
