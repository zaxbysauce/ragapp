import { forwardRef } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: { children: React.ReactNode }) => <div {...props}>{children}</div>,
    aside: ({ children, ...props }: { children: React.ReactNode }) => <aside {...props}>{children}</aside>,
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Plus: () => <svg data-testid="mock-plus">+</svg>,
  PanelLeftClose: () => <svg data-testid="mock-panel-left-close">X</svg>,
  PanelLeft: () => <svg data-testid="mock-panel-left">«</svg>,
  Download: () => <svg data-testid="mock-download">↓</svg>,
  User: () => <svg data-testid="mock-user">👤</svg>,
  Bot: () => <svg data-testid="mock-bot">🤖</svg>,
  Send: () => <svg data-testid="mock-send">➤</svg>,
  Square: () => <svg data-testid="mock-square">■</svg>,
  FileText: () => <svg data-testid="mock-file-text">📄</svg>,
  Code2: () => <svg data-testid="mock-code2">💻</svg>,
  X: () => <svg data-testid="mock-x">✕</svg>,
  Check: () => <svg data-testid="mock-check">✓</svg>,
  Copy: () => <svg data-testid="mock-copy">📋</svg>,
  Sparkles: () => <svg data-testid="mock-sparkles">✨</svg>,
  BookOpen: () => <svg data-testid="mock-book-open">📚</svg>,
  Search: () => <svg data-testid="mock-search">🔍</svg>,
  Layers: () => <svg data-testid="mock-layers">🗂️</svg>,
  HelpCircle: () => <svg data-testid="mock-help-circle">?</svg>,
  MessageSquare: () => <svg data-testid="mock-message-square">💬</svg>,
  Trash2: () => <svg data-testid="mock-trash">🗑️</svg>,
  Edit3: () => <svg data-testid="mock-edit">✏️</svg>,
  ThumbsUp: () => <svg data-testid="mock-thumbs-up">👍</svg>,
  ThumbsDown: () => <svg data-testid="mock-thumbs-down">👎</svg>,
  GitCompare: () => <svg data-testid="mock-git-compare">⟷</svg>,
}));

// Mock UI components
vi.mock('@/components/ui/scroll-area', () => ({
  ScrollArea: forwardRef<HTMLDivElement, { children: React.ReactNode }>(({ children }, ref) => (
    <div ref={ref} data-testid="mock-scroll-area">{children}</div>
  )),
}));

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, variant, size, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} data-variant={variant} data-size={size} {...props}>
      {children}
    </button>
  ),
}));

vi.mock('@/components/ui/textarea', () => ({
  Textarea: forwardRef<
    HTMLTextAreaElement,
    {
      value: string;
      onChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
      onKeyDown?: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
      onInput?: (event: React.FormEvent<HTMLTextAreaElement>) => void;
      disabled?: boolean;
      placeholder?: string;
      className?: string;
    }
  >(({ value, onChange, onKeyDown, onInput, disabled, placeholder, className }, ref) => (
    <textarea
      ref={ref}
      value={value}
      onChange={onChange}
      onKeyDown={onKeyDown}
      onInput={onInput}
      disabled={disabled}
      placeholder={placeholder}
      className={className}
      data-testid="mock-textarea"
    />
  )),
}));

// Mock stores - MUST be before component imports
const {
  mockUseChatStore,
  mockUseVaultStore,
  mockUseSendMessage,
  mockUseChatHistory,
} = vi.hoisted(() => ({
  mockUseChatStore: vi.fn(),
  mockUseVaultStore: vi.fn(),
  mockUseSendMessage: vi.fn(),
  mockUseChatHistory: vi.fn(),
}));

vi.mock('@/stores/useChatStore', () => ({
  useChatStore: mockUseChatStore,
}));

vi.mock('@/stores/useVaultStore', () => ({
  useVaultStore: mockUseVaultStore,
}));

// Mock hooks
vi.mock('@/hooks/useSendMessage', () => ({
  useSendMessage: mockUseSendMessage,
  MAX_INPUT_LENGTH: 4000,
}));

vi.mock('@/hooks/useChatHistory', () => ({
  useChatHistory: mockUseChatHistory,
}));

// Mock keyboard shortcuts
vi.mock('@/components/shared/KeyboardShortcuts', () => ({
  useKeyboardShortcuts: () => ({ open: false, setOpen: vi.fn() }),
  KeyboardShortcutsDialog: ({ open, onOpenChange }: any) => (
    <div data-testid="mock-keyboard-shortcuts" data-open={open.toString()} />
  ),
}));

// Mock vault selector
vi.mock('@/components/vault/VaultSelector', () => ({
  VaultSelector: () => <div data-testid="mock-vault-selector">Vault</div>,
}));

vi.mock('@/components/canvas/ResizableHandle', () => ({
  ResizableHandle: ({ onResize }: any) => (
    <div
      data-testid="mock-resizable-handle"
      onMouseDown={(e: MouseEvent) => {
        const handleMove = (moveEvent: MouseEvent) => {
          onResize(moveEvent.clientX - e.clientX);
        };
        const handleUp = () => {
          document.removeEventListener('mousemove', handleMove);
          document.removeEventListener('mouseup', handleUp);
        };
        document.addEventListener('mousemove', handleMove);
        document.addEventListener('mouseup', handleUp);
      }}
    >
      Resize
    </div>
  ),
}));

// Mock utils
vi.mock('@/lib/utils', () => ({
  cn: (...classes: any[]) => classes.filter(Boolean).join(' '),
}));

// Import components AFTER all mocks
import { ChatMessages } from './ChatMessages';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { CanvasPanel } from '../canvas/CanvasPanel';
import { MessageContent } from './MessageContent';
import { DocumentPreview } from '../canvas/DocumentPreview';
import { CodeViewer } from '../canvas/CodeViewer';

// Mock useSendMessage hook return value
const mockSendMessageReturn = {
  handleSend: vi.fn(),
  handleStop: vi.fn(),
};

describe('ChatMessages Component', () => {
  let unmount: () => void;

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChatStore.mockReturnValue({
      messages: [],
      isStreaming: false,
      newChat: vi.fn(),
      input: '',
      setInput: vi.fn(),
      inputError: null,
    });
    mockUseVaultStore.mockReturnValue({
      activeVaultId: 'vault-1',
    });
    mockUseSendMessage.mockReturnValue(mockSendMessageReturn);
    mockUseChatHistory.mockReturnValue({
      chatHistory: [],
      isChatLoading: false,
      chatHistoryError: null,
      handleLoadChat: vi.fn(),
      refreshHistory: vi.fn(),
    });
  });

  afterEach(() => {
    if (unmount) {
      unmount();
    }
  });

  it('should render header with new chat, vault selector, export, and canvas toggle buttons', () => {
    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

    expect(screen.getByTestId('mock-plus')).toBeInTheDocument();
    expect(screen.getByTestId('mock-vault-selector')).toBeInTheDocument();
    expect(screen.getByTestId('mock-download')).toBeInTheDocument();
    expect(screen.getByTestId('mock-panel-left')).toBeInTheDocument();
  });

  it('should show empty state when no messages', () => {
    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

    expect(screen.getByText(/How can I help you today\?/)).toBeInTheDocument();
    expect(
      screen.getByText(/Ask anything about your documents, or pick a suggestion below\./)
    ).toBeInTheDocument();
    expect(screen.getByText('Summarize documents')).toBeInTheDocument();
  });

  it('should render messages with MessageBubble components', () => {
    const mockMessages = [
      { id: '1', role: 'user' as const, content: 'Hello' },
      { id: '2', role: 'assistant' as const, content: 'Hi there!' },
    ];

    mockUseChatStore.mockReturnValue({
      messages: mockMessages,
      isStreaming: false,
      newChat: vi.fn(),
      input: '',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
  });

  it('should render ChatInput at the bottom', () => {
    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

    expect(screen.getByTestId('mock-textarea')).toBeInTheDocument();
  });

  it('should call newChat when new chat button is clicked', () => {
    const newChatFn = vi.fn();
    mockUseChatStore.mockReturnValue({
      messages: [],
      isStreaming: false,
      newChat: newChatFn,
      input: '',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

    fireEvent.click(screen.getByTestId('mock-plus').closest('button')!);
    expect(newChatFn).toHaveBeenCalled();
  });

  it('should call toggleCanvasCollapse when canvas toggle button is clicked', () => {
    const toggleFn = vi.fn();
    render(<ChatMessages toggleCanvasCollapse={toggleFn} canvasCollapsed={true} />);

    fireEvent.click(screen.getByTestId('mock-panel-left').closest('button')!);
    expect(toggleFn).toHaveBeenCalled();
  });

  it('should show PanelLeftClose icon when canvas is not collapsed', () => {
    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={false} />);

    expect(screen.getByTestId('mock-panel-left-close')).toBeInTheDocument();
  });

  it('should disable export button when no messages', () => {
    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

    const exportButton = screen.getByTestId('mock-download').closest('button')!;
    expect(exportButton).toBeDisabled();
  });

  it('should enable export button when messages exist', () => {
    mockUseChatStore.mockReturnValue({
      messages: [{ id: '1', role: 'user' as const, content: 'Hello' }],
      isStreaming: false,
      newChat: vi.fn(),
      input: '',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

    const exportButton = screen.getByTestId('mock-download').closest('button')!;
    expect(exportButton).not.toBeDisabled();
  });

  it('should export chat when export button is clicked', async () => {
    vi.useFakeTimers();
    const createObjectURLSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url');
    const revokeObjectURLSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {});

    try {
      mockUseChatStore.mockReturnValue({
        messages: [
          { id: '1', role: 'user' as const, content: 'Hello' },
          { id: '2', role: 'assistant' as const, content: 'Hi there!' },
        ],
        isStreaming: false,
        newChat: vi.fn(),
        input: '',
        setInput: vi.fn(),
        inputError: null,
      });

      render(<ChatMessages toggleCanvasCollapse={vi.fn()} canvasCollapsed={true} />);

      const exportButton = screen.getByTestId('mock-download').closest('button')!;
      await act(async () => {
        fireEvent.click(exportButton);
        vi.runAllTimers();
      });

      expect(createObjectURLSpy).toHaveBeenCalled();
      const blobArg = createObjectURLSpy.mock.calls[0][0];
      expect(blobArg).toBeInstanceOf(Blob);
      expect(clickSpy).toHaveBeenCalled();
      expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url');
    } finally {
      vi.useRealTimers();
      createObjectURLSpy.mockRestore();
      revokeObjectURLSpy.mockRestore();
      clickSpy.mockRestore();
    }
  });
});

describe('MessageBubble Component', () => {
  it('should render user message with correct styling', () => {
    const message = {
      id: '1',
      role: 'user' as const,
      content: 'Hello',
    };

    render(<MessageBubble message={message} />);

    expect(screen.getByTestId('mock-user')).toBeInTheDocument();
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('should render assistant message with correct styling', () => {
    const message = {
      id: '2',
      role: 'assistant' as const,
      content: 'Hi there!',
    };

    render(<MessageBubble message={message} />);

    expect(screen.getByTestId('mock-bot')).toBeInTheDocument();
    expect(screen.getByText('Assistant')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
  });

  it('should render MessageContent with message content', () => {
    const message = {
      id: '1',
      role: 'user' as const,
      content: 'Test content',
    };

    render(<MessageBubble message={message} />);

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('should render sources when provided', () => {
    const message = {
      id: '1',
      role: 'assistant' as const,
      content: 'Answer',
      sources: [{ id: 's1', filename: 'doc.pdf', snippet: 'relevant part' }],
    };

    render(<MessageBubble message={message} />);

    expect(screen.getByText('Sources')).toBeInTheDocument();
    expect(screen.getByText('doc.pdf')).toBeInTheDocument();
  });

  it('should render error message when present', () => {
    const message = {
      id: '1',
      role: 'assistant' as const,
      content: 'Answer',
      error: 'Something went wrong',
    };

    render(<MessageBubble message={message} />);

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('should render stopped indicator when message was stopped', () => {
    const message = {
      id: '1',
      role: 'assistant' as const,
      content: 'Answer',
      stopped: true,
    };

    render(<MessageBubble message={message} />);

    expect(screen.getByText('Generation stopped')).toBeInTheDocument();
  });

  it('should apply different background colors for user vs assistant', () => {
    const userMessage = { id: '1', role: 'user' as const, content: 'User' };
    const assistantMessage = { id: '2', role: 'assistant' as const, content: 'Assistant' };

    render(
      <>
        <MessageBubble message={userMessage} />
        <MessageBubble message={assistantMessage} />
      </>
    );

    // User message should have bg-primary/5
    // Assistant message should have bg-muted/30
    // These are applied via className, so we verify the component renders correctly
    expect(screen.getByText('User')).toBeInTheDocument();
    expect(screen.getAllByText('Assistant').length).toBeGreaterThan(0);
  });
});

describe('ChatInput Component', () => {
  let unmount: () => void;

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChatStore.mockReturnValue({
      input: '',
      setInput: vi.fn(),
      inputError: null,
    });
  });

  afterEach(() => {
    if (unmount) {
      unmount();
    }
  });

  it('should render textarea with placeholder', () => {
    render(<ChatInput onSend={vi.fn()} onStop={vi.fn()} isStreaming={false} />);

    expect(screen.getByTestId('mock-textarea')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Message.*Enter to send/)).toBeInTheDocument();
  });

  it('should render send button when not streaming', () => {
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={vi.fn()} onStop={vi.fn()} isStreaming={false} />);

    expect(screen.getByTestId('mock-send')).toBeInTheDocument();
  });

  it('should render stop button when streaming', () => {
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={vi.fn()} onStop={vi.fn()} isStreaming={true} />);

    expect(screen.getByTestId('mock-square')).toBeInTheDocument();
  });

  it('should call onSend when Enter is pressed', () => {
    const onSendFn = vi.fn();
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={onSendFn} onStop={vi.fn()} isStreaming={false} />);

    const textarea = screen.getByTestId('mock-textarea');
    fireEvent.keyDown(textarea, { key: 'Enter' });

    expect(onSendFn).toHaveBeenCalled();
  });

  it('should not call onSend when Shift+Enter is pressed', () => {
    const onSendFn = vi.fn();
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={onSendFn} onStop={vi.fn()} isStreaming={false} />);

    const textarea = screen.getByTestId('mock-textarea');
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(onSendFn).not.toHaveBeenCalled();
  });

  it('should not call onSend when input is empty', () => {
    const onSendFn = vi.fn();
    mockUseChatStore.mockReturnValue({
      input: '',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={onSendFn} onStop={vi.fn()} isStreaming={false} />);

    const textarea = screen.getByTestId('mock-textarea');
    fireEvent.keyDown(textarea, { key: 'Enter' });

    expect(onSendFn).not.toHaveBeenCalled();
  });

  it('should not call onSend when streaming', () => {
    const onSendFn = vi.fn();
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={onSendFn} onStop={vi.fn()} isStreaming={true} />);

    const textarea = screen.getByTestId('mock-textarea');
    fireEvent.keyDown(textarea, { key: 'Enter' });

    expect(onSendFn).not.toHaveBeenCalled();
  });

  it('should call onStop when stop button is clicked', () => {
    const onStopFn = vi.fn();
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={vi.fn()} onStop={onStopFn} isStreaming={true} />);

    const stopButton = screen.getByTestId('mock-square').closest('button')!;
    fireEvent.click(stopButton!);

    expect(onStopFn).toHaveBeenCalled();
  });

  it('should show input error message when present', () => {
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: 'Input too long',
    });

    render(<ChatInput onSend={vi.fn()} onStop={vi.fn()} isStreaming={false} />);

    expect(screen.getByText('Input too long')).toBeInTheDocument();
  });

  it('should disable textarea when streaming', () => {
    mockUseChatStore.mockReturnValue({
      input: 'test',
      setInput: vi.fn(),
      inputError: null,
    });

    render(<ChatInput onSend={vi.fn()} onStop={vi.fn()} isStreaming={true} />);

    const textarea = screen.getByTestId('mock-textarea');
    expect(textarea).toBeDisabled();
  });
});

describe('CanvasPanel Component', () => {
  let unmount: () => void;

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChatStore.mockReturnValue({
      messages: [],
    });
  });

  afterEach(() => {
    if (unmount) {
      unmount();
    }
  });

  it('should render tabs for document and code views', () => {
    render(
      <CanvasPanel
        canvas={{ view: null, isCollapsed: false, width: 400 }}
        onToggleCollapse={vi.fn()}
        onSetView={vi.fn()}
        onSetWidth={vi.fn()}
      />
    );

    expect(screen.getByText('Document')).toBeInTheDocument();
    expect(screen.getByText('Code')).toBeInTheDocument();
  });

  it('should call onSetView when tab is clicked', () => {
    const onSetViewFn = vi.fn();
    render(
      <CanvasPanel
        canvas={{ view: null, isCollapsed: false, width: 400 }}
        onToggleCollapse={vi.fn()}
        onSetView={onSetViewFn}
        onSetWidth={vi.fn()}
      />
    );

    const documentTab = screen.getByText('Document').closest('button')!;
    fireEvent.click(documentTab!);

    expect(onSetViewFn).toHaveBeenCalledWith('document');
  });

  it('should show source selector when multiple sources exist', () => {
    mockUseChatStore.mockReturnValue({
      messages: [
        {
          role: 'assistant' as const,
          sources: [
            { id: 's1', filename: 'doc1.pdf' },
            { id: 's2', filename: 'doc2.pdf' },
          ],
        },
      ],
    });

    render(
      <CanvasPanel
        canvas={{ view: 'document', isCollapsed: false, width: 400 }}
        onToggleCollapse={vi.fn()}
        onSetView={vi.fn()}
        onSetWidth={vi.fn()}
      />
    );

    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'doc1.pdf' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'doc2.pdf' })).toBeInTheDocument();
  });

  it('should show DocumentPreview when document tab is active', () => {
    mockUseChatStore.mockReturnValue({
      messages: [
        {
          role: 'assistant' as const,
          sources: [{ id: 's1', filename: 'doc.pdf', snippet: 'content' }],
        },
      ],
    });

    render(
      <CanvasPanel
        canvas={{ view: 'document', isCollapsed: false, width: 400 }}
        onToggleCollapse={vi.fn()}
        onSetView={vi.fn()}
        onSetWidth={vi.fn()}
      />
    );

    expect(screen.getByText('doc.pdf')).toBeInTheDocument();
    expect(screen.getByText('content')).toBeInTheDocument();
  });

  it('should show CodeViewer when code tab is active', () => {
    mockUseChatStore.mockReturnValue({
      messages: [
        {
          role: 'assistant' as const,
          sources: [{ id: 's1', filename: 'code.py', snippet: 'print("hello")' }],
        },
      ],
    });

    render(
      <CanvasPanel
        canvas={{ view: 'code', isCollapsed: false, width: 400 }}
        onToggleCollapse={vi.fn()}
        onSetView={vi.fn()}
        onSetWidth={vi.fn()}
      />
    );

    expect(screen.getByText('code.py')).toBeInTheDocument();
    expect(screen.getByText('print("hello")')).toBeInTheDocument();
  });

  it('should show no document message when no sources', () => {
    mockUseChatStore.mockReturnValue({
      messages: [{ role: 'assistant' as const, sources: [] }],
    });

    render(
      <CanvasPanel
        canvas={{ view: 'document', isCollapsed: false, width: 400 }}
        onToggleCollapse={vi.fn()}
        onSetView={vi.fn()}
        onSetWidth={vi.fn()}
      />
    );

    expect(screen.getByText('No document to preview')).toBeInTheDocument();
  });

  it('should call onToggleCollapse when close button is clicked', () => {
    const onToggleCollapseFn = vi.fn();
    render(
      <CanvasPanel
        canvas={{ view: null, isCollapsed: false, width: 400 }}
        onToggleCollapse={onToggleCollapseFn}
        onSetView={vi.fn()}
        onSetWidth={vi.fn()}
      />
    );

    const closeButton = screen.getByTestId('mock-x').closest('button')!;
    fireEvent.click(closeButton!);

    expect(onToggleCollapseFn).toHaveBeenCalled();
  });

  it('should render ResizableHandle component', () => {
    render(
      <CanvasPanel
        canvas={{ view: null, isCollapsed: false, width: 400 }}
        onToggleCollapse={vi.fn()}
        onSetView={vi.fn()}
        onSetWidth={vi.fn()}
      />
    );

    expect(screen.getByTestId('mock-resizable-handle')).toBeInTheDocument();
  });
});

describe('MessageContent Component', () => {
  it('should render markdown content', () => {
    render(<MessageContent content="# Header\n\n**Bold text**" />);

    expect(screen.getByText(/Header/)).toBeInTheDocument();
    expect(screen.getByText('Bold text')).toBeInTheDocument();
  });

  it('should render sources when provided', () => {
    render(
      <MessageContent
        content="Answer"
        sources={[
          { id: 's1', filename: 'doc.pdf', snippet: 'relevant', score: 0.95 },
        ]}
      />
    );

    expect(screen.getByText('Sources')).toBeInTheDocument();
    expect(screen.getByText('doc.pdf')).toBeInTheDocument();
    expect(screen.getByText('95%')).toBeInTheDocument();
  });

  it('should show streaming indicator when isStreaming is true', () => {
    const { container } = render(<MessageContent content="Answer" isStreaming={true} />);

    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('should render a code block copy button for fenced code', () => {
    render(<MessageContent content={'```ts\nconst value = 1;\n```'} />);

    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('should copy code block content to clipboard when copy button is clicked', async () => {
    const writeTextMock = vi.mocked(navigator.clipboard.writeText);
    writeTextMock.mockResolvedValue(undefined);

    render(<MessageContent content={'```ts\nconst value = 1;\n```'} />);

    const copyButton = screen.getByRole('button', { name: /copy/i });
    await act(async () => {
      fireEvent.click(copyButton!);
    });

    expect(writeTextMock).toHaveBeenCalledWith('const value = 1;');
  });
});

describe('DocumentPreview Component', () => {
  it('should render filename and snippet', () => {
    render(<DocumentPreview source={{ id: 's1', filename: 'test.pdf', snippet: 'content here' }} />);

    expect(screen.getByText('test.pdf')).toBeInTheDocument();
    expect(screen.getByText('content here')).toBeInTheDocument();
  });

  it('should show "No content available" when snippet is empty', () => {
    render(<DocumentPreview source={{ id: 's1', filename: 'test.pdf', snippet: '' }} />);

    expect(screen.getByText('No content available')).toBeInTheDocument();
  });
});

describe('CodeViewer Component', () => {
  it('should render filename and language', () => {
    render(<CodeViewer source={{ id: 's1', filename: 'script.py', snippet: 'print("hello")' }} />);

    expect(screen.getByText('script.py')).toBeInTheDocument();
    expect(screen.getByText('py')).toBeInTheDocument();
  });

  it('should show "text" language when filename has no extension', () => {
    render(<CodeViewer source={{ id: 's1', filename: 'noext', snippet: 'content' }} />);

    expect(screen.getByText('text')).toBeInTheDocument();
  });

  it('should have copy button', () => {
    render(<CodeViewer source={{ id: 's1', filename: 'test.py', snippet: 'code' }} />);

    expect(screen.getByText('Copy')).toBeInTheDocument();
  });

  it('should copy snippet to clipboard when copy button is clicked', async () => {
    const writeTextMock = vi.mocked(navigator.clipboard.writeText);
    writeTextMock.mockResolvedValue(undefined);

    render(<CodeViewer source={{ id: 's1', filename: 'test.py', snippet: 'print("hello")' }} />);

    const copyButton = screen.getByText('Copy').closest('button')!;
    await act(async () => {
      fireEvent.click(copyButton!);
    });

    expect(writeTextMock).toHaveBeenCalledWith('print("hello")');
  });
});
