# Chat UI Redesign Scaffold

## Tech Stack
- React 18 + TypeScript + Vite
- Tailwind CSS + shadcn/ui
- Zustand (state management)
- Framer Motion (animations)
- react-markdown + remark-gfm
- Lucide React

## File Structure
```
frontend/
├── src/
│   ├── pages/
│   │   └── ChatPageRedesigned.tsx
│   ├── stores/
│   │   └── useChatStore.ts
│   ├── hooks/
│   │   ├── useSendMessage.ts
│   │   └── useChatHistory.ts
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatMessages.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   └── ChatInput.tsx
│   │   ├── canvas/
│   │   │   ├── CanvasPanel.tsx
│   │   │   ├── DocumentPreview.tsx
│   │   │   ├── CodeViewer.tsx
│   │   │   └── ResizableHandle.tsx
│   │   └── shared/
│   │       └── MessageContent.tsx
│   ├── lib/
│   │   ├── api.ts
│   │   └── utils.ts
│   ├── types/
│   │   └── index.ts
│   └── index.css
├── package.json
└── tailwind.config.js
```

## Key Features
- Two-pane resizable layout with collapsible canvas
- Multiple canvas tabs (Document, Code)
- Auto-display latest assistant sources
- Live document editing simulation
- Streaming responses with typing cursor
- Full theme support (light/dark/system)
- Clipboard copy, export, keyboard shortcuts
- Source cards with relevance scores

See following files for full implementation.