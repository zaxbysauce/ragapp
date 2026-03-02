# Chat UI Redesign - Implementation Guide

## Overview
This is a complete, production-ready scaffold for a redesigned AI chat interface inspired by ChatGPT, Claude, and Gemini. It features a two-pane layout with a resizable canvas for document/code previews.

## Key Features Implemented

### 1. Two-Pane Layout
- Main chat area (2/3 width by default)
- Resizable right-side canvas (collapsible)
- Smooth resize with drag handle
- Persistent width preference (via Zustand persist)

### 2. Canvas Panel
- **Multiple tabs**: Document Preview & Code Viewer
- **Auto-population**: Shows latest assistant response sources automatically
- **Live editing simulation**: Content morphs to final version (replace with real diffs from API)
- **Collapsible**: Toggle button in header
- **Resizable**: Drag handle between panes

### 3. Chat Features
- Streaming responses with typing cursor
- User/assistant message bubbles with avatars
- Markdown rendering with syntax highlighting
- Copy message button (appears on hover)
- Auto-resizing textarea input
- Enter to send, Shift+Enter for newline
- Stop generation button
- Timestamps on messages

### 4. State Management
- Zustand store with persistence
- Separate chat sessions support
- Canvas state management
- Theme system (light/dark/system)

## File Structure
```
frontend/
├── src/
│   ├── pages/
│   │   └── ChatPageRedesigned.tsx    # Main page component
│   ├── stores/
│   │   └── useChatStore.ts          # Zustand store
│   ├── hooks/
│   │   └── useSendMessage.ts        # Message streaming hook
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatMessages.tsx     # Message list + input
│   │   │   ├── MessageBubble.tsx    # Individual message
│   │   │   └── ChatInput.tsx        # Input textarea
│   │   ├── canvas/
│   │   │   ├── CanvasPanel.tsx      # Main canvas container
│   │   │   ├── DocumentPreview.tsx  # Document viewer
│   │   │   ├── CodeViewer.tsx       # Code viewer
│   │   │   └── ResizableHandle.tsx  # Drag handle
│   │   ├── shared/
│   │   │   └── MessageContent.tsx   # Markdown renderer
│   │   ├── ui/                      # shadcn/ui components
│   │   └── theme-provider.tsx
│   ├── lib/
│   │   ├── api.ts                   # API streaming function
│   │   └── utils.ts                 # cn() utility
│   └── types/
│       └── index.ts                 # TypeScript types
├── components.json                  # shadcn/ui config
├── package.json
├── tailwind.config.js
├── vite.config.ts
└── tsconfig.json
```

## Integration Steps

### 1. Install Dependencies
```bash
cd frontend
npm install
```

Required packages are already in package.json:
- React 18 + TypeScript + Vite
- Tailwind CSS
- shadcn/ui components (Radix primitives)
- Zustand
- Framer Motion
- react-markdown + remark-gfm
- next-themes
- lucide-react
- clsx + tailwind-merge
- class-variance-authority

### 2. Set Up shadcn/ui
```bash
npx shadcn-ui@latest init
npx shadcn-ui@latest add button
npx shadcn-ui@latest add textarea
npx shadcn-ui@latest add scroll-area
```

This will install Radix UI dependencies and update components.json.

### 3. Update API Endpoint
In `src/lib/api.ts`, modify the `chatStream` function to match your existing backend endpoint:
```typescript
const response = await fetch("/api/chat", {  // Change to your endpoint
  method: "POST",
  ...
});
```

Ensure the streaming format matches your backend's SSE format. The current implementation expects:
```
data: {"type":"chunk","content":"..."}
data: {"type":"sources","sources":[...]}
data: {"type":"error","message":"..."}
data: [DONE]
```

### 4. Connect Zustand Store to Backend
The store currently has placeholder logic. Connect it to your actual API by:
- Updating `useSendMessage.ts` to use your authentication headers if needed
- Integrating with your existing `useChatHistory.ts` for loading past conversations
- Mapping your backend's message format to the `Message` interface

### 5. Live Document Editing
The `DocumentPreview` and `CodeViewer` components simulate live edits with a simple interpolation. To implement real-time edits:
- Listen for WebSocket or SSE events from your backend
- Update `source.content` in the store as edits arrive
- The components will automatically re-render with updated content

### 6. Source Selection
The canvas currently auto-selects the first source. To implement manual selection:
- Add a source list component that displays all available sources
- Update `canvas.activeSourceId` in the store when user selects a source
- Consider adding a source selector dropdown in the canvas header

### 7. Theme System
The design uses CSS variables with light/dark mode via `next-themes`. Your existing theme system in `index.css` should be compatible. The `ThemeProvider` component wraps the app.

### 8. Replace Frontend Routing
If you have an existing router (React Router), replace your current chat page route to render `ChatPageRedesigned` instead of the old component. Keep your existing layout/higher-order components.

### 9. Styling Customization
The design uses a neutral palette with subtle accent colors. To customize:
- Update CSS variables in `index.css`
- Modify Tailwind config for brand colors
- Adjust shadcn/ui variants in `components.json`

### 10. Testing
- Test resizing behavior with different screen sizes
- Verify streaming responses render correctly
- Check theme switching (light/dark/system)
- Ensure mobile responsiveness (currently desktop-optimized)

## Important Notes

### Current Assumptions
- Two-pane layout with right canvas
- Canvas shows assistant sources only
- Sources array is present in assistant messages
- Streaming via SSE in the format shown above
- No file uploads (as requested)
- Desktop-first design

### To Add Later
- Mobile responsive layout (hamburger menu, stacked panes)
- Source selector component
- Real-time document editing via WebSocket
- Code syntax highlighting (consider prism.js or highlight.js)
- Export conversations
- Search in chat history
- Settings panel

## TypeScript Interfaces

The scaffold uses these core types (in `src/types/index.ts`):
- `Message`: Chat message with role, content, sources, timestamps
- `Source`: Document/code reference with filename, snippet, score, content, language
- `ChatSession`: Conversation with messages and metadata
- `CanvasState`: View type (document/code), collapse state, width, active source

## Migration Path

1. Copy the entire `src/` directory into your existing frontend project
2. Install missing dependencies from package.json
3. Replace your current chat page with `ChatPageRedesigned`
4. Gradually migrate state management useChatStore to your existing store patterns
5. Point API calls to your backend
6. Customize styling to match your brand
7. Add any missing features from your current chat (e.g., vault selector)

## Performance Considerations

- Streaming uses React's concurrent rendering patterns
- Canvas contents are re-rendered only when active source changes
- Resizable handle uses minimal events
- Zustand store is persisted to localStorage (canvas width, theme)

## Known Limitations

- No mobile optimization (needs media queries)
- No source selection UI (auto-uses first source)
- Live editing is simulated (needs real diff updates)
- No code syntax highlighting
- No export functionality
- No search in history

## Support

This scaffold is designed to be easily adapted. All components are:
- Fully typed with TypeScript
- Self-contained with minimal dependencies
- Following shadcn/ui patterns for consistency
- Using Tailwind CSS for easy styling overrides

For questions, refer to the file comments or the shadcn/ui documentation.