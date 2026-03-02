"use client";

import React, { useCallback, useRef, useState } from "react";
import { GripVertical } from "lucide-react";

interface ResizableHandleProps {
  onResize: (width: number) => void;
  minWidth?: number;
  maxWidth?: number;
}

export function ResizableHandle({ onResize, minWidth = 300, maxWidth = 800 }: ResizableHandleProps) {
  const [isDragging, setIsDragging] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX;
    startWidthRef.current = e.currentTarget.parentElement?.parentElement?.getBoundingClientRect().width || 400;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const delta = moveEvent.clientX - startXRef.current;
      const newWidth = startWidthRef.current - delta;
      onResize(Math.max(minWidth, Math.min(maxWidth, newWidth)));
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  }, [onResize, minWidth, maxWidth]);

  return (
    <div
      className={`relative flex items-center justify-center cursor-col-resize transition-colors ${
        isDragging ? "bg-accent" : "hover:bg-accent/50"
      }`}
      onMouseDown={handleMouseDown}
    >
      <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-1 group-hover:w-2 transition-all" />
      <GripVertical className={`h-6 w-4 text-muted-foreground ${isDragging ? "rotate-90" : ""}`} />
    </div>
  );
}