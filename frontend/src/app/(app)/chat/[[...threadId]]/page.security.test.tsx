import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import '@testing-library/jest-dom';

// ============================================================================
// ADVERSARIAL SECURITY TESTS - Malformed inputs, oversized payloads, 
// injection attempts, navigation attacks, boundary violations
// ============================================================================

// Hoisted mocks
const mocks = vi.hoisted(() => {
  const mockLoadSession = vi.fn().mockResolvedValue(undefined);
  const mockClearSession = vi.fn();
  const mockSetInput = vi.fn();
  const mockNavigate = vi.fn();
  const mockCreateChatSession = vi.fn();
  const mockGetActiveVault = vi.fn();
  const mockUseParams = vi.fn().mockReturnValue({ threadId: undefined });
  
  return {
    mockLoadSession,
    mockClearSession,
    mockSetInput,
    mockNavigate,
    mockCreateChatSession,
    mockGetActiveVault,
    mockUseParams,
    useAuthStoreMock: vi.fn(() => ({
      mustChangePassword: false,
    })),
    useVaultStoreMock: vi.fn(() => ({
      activeVaultId: 1,
      getActiveVault: mockGetActiveVault,
    })),
    useChatSessionMock: vi.fn(() => ({
      isLoading: false,
      loadSession: mockLoadSession,
      messages: [],
      clearSession: mockClearSession,
    })),
  };
});

// Mock react-router-dom
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mocks.mockNavigate,
    useParams: mocks.mockUseParams,
  };
});

// Mock stores
vi.mock('@/stores/authStore', () => ({
  useAuthStore: mocks.useAuthStoreMock,
}));

vi.mock('@/stores/useVaultStore', () => ({
  useVaultStore: mocks.useVaultStoreMock,
}));

vi.mock('@/stores/useChatStore', () => ({
  useChatStore: Object.assign(
    vi.fn(() => ({
      setInput: mocks.mockSetInput,
    })),
    {
      getState: () => ({
        setInput: mocks.mockSetInput,
      }),
    }
  ),
}));

vi.mock('@/components/chat/useChatSession', () => ({
  useChatSession: mocks.useChatSessionMock,
}));

vi.mock('@/lib/api', () => ({
  createChatSession: mocks.mockCreateChatSession,
}));

vi.mock('@/lib/auth', () => ({
  useRequirePasswordChange: vi.fn(),
}));

// Comprehensive lucide-react mock with all icons
vi.mock('lucide-react', () => ({
  Loader2: ({ className }: { className?: string }) => (
    <span data-testid="mock-loader" className={className}>⟳</span>
  ),
  Plus: ({ className }: { className?: string }) => (
    <span data-testid="mock-plus" className={className}>+</span>
  ),
  PanelRight: ({ className }: { className?: string }) => (
    <span data-testid="mock-panel-right" className={className}>→</span>
  ),
  PanelRightClose: ({ className }: { className?: string }) => (
    <span data-testid="mock-panel-right-close" className={className}>×</span>
  ),
  User: () => <span data-testid="mock-user">👤</span>,
  Bot: () => <span data-testid="mock-bot">🤖</span>,
  Send: () => <span data-testid="mock-send">➤</span>,
  Square: () => <span data-testid="mock-square">■</span>,
  ChevronDown: () => <span data-testid="mock-chevron-down">▼</span>,
  Check: () => <span data-testid="mock-check">✓</span>,
  X: () => <span data-testid="mock-x">✕</span>,
  Menu: () => <span data-testid="mock-menu">☰</span>,
  Settings: () => <span data-testid="mock-settings">⚙</span>,
  FileText: () => <span data-testid="mock-file-text">📄</span>,
  Upload: () => <span data-testid="mock-upload">⬆</span>,
  Trash: () => <span data-testid="mock-trash">🗑</span>,
  Copy: () => <span data-testid="mock-copy">📋</span>,
  ExternalLink: () => <span data-testid="mock-external-link">🔗</span>,
  MessageSquare: () => <span data-testid="mock-message-square">💬</span>,
  Search: () => <span data-testid="mock-search">🔍</span>,
  AlertCircle: () => <span data-testid="mock-alert-circle">⚠</span>,
  Info: () => <span data-testid="mock-info">ℹ</span>,
  RefreshCw: () => <span data-testid="mock-refresh-cw">🔄</span>,
  ZoomIn: () => <span data-testid="mock-zoom-in">🔍+</span>,
  ZoomOut: () => <span data-testid="mock-zoom-out">🔍-</span>,
}));

vi.mock('@/components/ui/scroll-area', () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-scroll-area">{children}</div>
  ),
}));

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, variant, size, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} data-variant={variant} data-size={size} {...props}>
      {children}
    </button>
  ),
}));

vi.mock('@/components/ui/select', () => ({
  Select: ({ children }: any) => <div data-testid="mock-select">{children}</div>,
  SelectTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  SelectValue: ({ children }: any) => <span>{children}</span>,
}));

vi.mock('@/components/ui/dropdown-menu', () => ({
  DropdownMenu: ({ children }: any) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: any) => <div>{children}</div>,
  DropdownMenuContent: ({ children }: any) => <div>{children}</div>,
  DropdownMenuItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  DropdownMenuSeparator: () => <hr />,
}));

vi.mock('@/components/ui/tooltip', () => ({
  TooltipProvider: ({ children }: any) => <div>{children}</div>,
  Tooltip: ({ children }: any) => <div>{children}</div>,
  TooltipTrigger: ({ children }: any) => <div>{children}</div>,
  TooltipContent: ({ children }: any) => <div>{children}</div>,
}));

vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: { children: React.ReactNode }) => <div {...props}>{children}</div>,
    p: ({ children, ...props }: { children: React.ReactNode }) => <p {...props}>{children}</p>,
    button: ({ children, ...props }: { children: React.ReactNode }) => <button {...props}>{children}</button>,
    span: ({ children, ...props }: { children: React.ReactNode }) => <span {...props}>{children}</span>,
    li: ({ children, ...props }: { children: React.ReactNode }) => <li {...props}>{children}</li>,
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/chat/EmptyTranscript', () => ({
  EmptyTranscript: ({ onPromptClick, onUploadClick, hasDocuments }: any) => (
    <div data-testid="mock-empty-transcript">
      <button data-testid="prompt-btn" onClick={() => onPromptClick?.('Test prompt')}>
        Test Prompt
      </button>
      <button data-testid="xss-prompt-btn" onClick={() => onPromptClick?.('<script>alert(1)</script>')}>
        XSS Prompt
      </button>
      <button data-testid="template-prompt-btn" onClick={() => onPromptClick?.('${alert(1)}')}>
        Template Injection
      </button>
      <button data-testid="unicode-prompt-btn" onClick={() => onPromptClick?.('\u0000\u202E\uFEFF')}>
        Unicode Attack
      </button>
      <button data-testid="emoji-prompt-btn" onClick={() => onPromptClick?.('💣💥🔥😈')}>
        Emoji Bomb
      </button>
      <button data-testid="sql-prompt-btn" onClick={() => onPromptClick?.("'; DROP TABLE users; --")}>
        SQL Injection
      </button>
      <button data-testid="html-prompt-btn" onClick={() => onPromptClick?.('<img src=x onerror=alert(1)>')}>
        HTML Injection
      </button>
      <button data-testid="oversized-prompt-btn" onClick={() => onPromptClick?.('A'.repeat(100000))}>
        Oversized Payload
      </button>
      {!hasDocuments && (
        <button data-testid="upload-btn" onClick={onUploadClick}>
          Upload Documents
        </button>
      )}
    </div>
  ),
}));

vi.mock('@/components/vault/VaultSelector', () => ({
  VaultSelector: () => <div data-testid="mock-vault-selector">Vault</div>,
}));

vi.mock('@/components/chat/ChatThread', () => ({
  ChatThread: ({ className }: { className?: string }) => (
    <div data-testid="mock-chat-thread" className={className}>ChatThread</div>
  ),
}));

import ChatPage from './page';

const renderWithRouter = (initialEntries: string[]) => {
  return render(
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:threadId" element={<ChatPage />} />
        <Route path="/documents" element={<div data-testid="documents-page">Documents Page</div>} />
        <Route path="/settings" element={<div data-testid="settings-page">Settings Page</div>} />
      </Routes>
    </MemoryRouter>
  );
};

describe('ChatPage Adversarial Security Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    
    mocks.useAuthStoreMock.mockReturnValue({ mustChangePassword: false });
    mocks.useVaultStoreMock.mockReturnValue({
      activeVaultId: 1,
      getActiveVault: mocks.mockGetActiveVault.mockReturnValue({ name: 'Test Vault', file_count: 5 }),
    });
    mocks.useChatSessionMock.mockReturnValue({
      isLoading: false,
      loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
      messages: [],
      clearSession: mocks.mockClearSession,
    });
    mocks.mockCreateChatSession.mockResolvedValue({ id: 123, title: 'New Chat' });
    mocks.mockGetActiveVault.mockReturnValue({ name: 'Test Vault', file_count: 5 });
    mocks.mockUseParams.mockReturnValue({ threadId: undefined });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ==========================================================================
  // CATEGORY 1: MALFORMED THREAD ID INPUTS
  // ==========================================================================
  describe('Malformed threadId attacks', () => {
    it('SEC-001: Rejects NaN threadId - does not call loadSession', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: 'not-a-number' });
      
      renderWithRouter(['/chat/not-a-number']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt returns NaN for non-numeric strings, so loadSession should NOT be called
      expect(mocks.mockLoadSession).not.toHaveBeenCalled();
    });

    it('SEC-002: Accepts negative threadId - backend validation required', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '-1' });
      
      renderWithRouter(['/chat/-1']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt returns -1, so loadSession IS called
      // Backend must validate negative session IDs
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(-1);
    });

    it('SEC-003: Truncates floating point threadId to integer', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '42.99' });
      
      renderWithRouter(['/chat/42.99']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt truncates to 42
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(42);
    });

    it('SEC-004: Handles large threadId without crash', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: String(Number.MAX_SAFE_INTEGER) });
      
      // Should not crash
      expect(() => {
        renderWithRouter([`/chat/${Number.MAX_SAFE_INTEGER}`]);
      }).not.toThrow();
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Should attempt to load (backend will reject)
      expect(mocks.mockLoadSession).toHaveBeenCalled();
    });

    it('SEC-005: ThreadId zero calls loadSession(0) - backend must reject', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '0' });
      
      renderWithRouter(['/chat/0']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt('0') = 0, isNaN(0) = false, so loadSession IS called with 0
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(0);
    });

    it('SEC-006: Handles large negative threadId without crash', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '-999999999' });
      
      expect(() => {
        renderWithRouter(['/chat/-999999999']);
      }).not.toThrow();
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      expect(mocks.mockLoadSession).toHaveBeenCalled();
    });

    it('SEC-007: Scientific notation threadId - parseInt truncates', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '1e5' });
      
      renderWithRouter(['/chat/1e5']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt('1e5') = 1 (stops at non-digit)
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(1);
    });
  });

  // ==========================================================================
  // CATEGORY 2: OVERSIZED PAYLOAD ATTACKS
  // ==========================================================================
  describe('Oversized payload attacks', () => {
    it('SEC-008: Handles 100KB prompt - setInput called with full content', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const oversizedButton = screen.getByTestId('oversized-prompt-btn');
      await userEvent.click(oversizedButton);
      
      // Verify the oversized payload is passed through
      expect(mocks.mockSetInput).toHaveBeenCalledWith('A'.repeat(100000));
    });

    it('SEC-009: Handles Unicode null byte attack in prompt', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const unicodeButton = screen.getByTestId('unicode-prompt-btn');
      await userEvent.click(unicodeButton);
      
      // Null bytes and special Unicode should be passed through
      const calledValue = mocks.mockSetInput.mock.calls[0]?.[0];
      expect(calledValue).toContain('\x00');
      expect(calledValue).toContain('\u202E'); // RTL override
      expect(calledValue).toContain('\uFEFF'); // BOM
    });

    it('SEC-010: Handles emoji bomb attack', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const emojiButton = screen.getByTestId('emoji-prompt-btn');
      await userEvent.click(emojiButton);
      
      // Emoji should be passed through
      expect(mocks.mockSetInput).toHaveBeenCalledWith('💣💥🔥😈');
    });
  });

  // ==========================================================================
  // CATEGORY 3: INJECTION ATTACKS
  // ==========================================================================
  describe('Injection attack attempts', () => {
    it('SEC-011: Handles XSS payload in prompt - passes through to store', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const xssButton = screen.getByTestId('xss-prompt-btn');
      await userEvent.click(xssButton);
      
      // The component passes the raw prompt to setInput
      // Sanitization happens downstream when rendering
      expect(mocks.mockSetInput).toHaveBeenCalledWith('<script>alert(1)</script>');
    });

    it('SEC-012: Handles template literal injection in prompt', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const templateButton = screen.getByTestId('template-prompt-btn');
      await userEvent.click(templateButton);
      
      // Template literal injection attempt
      expect(mocks.mockSetInput).toHaveBeenCalledWith('${alert(1)}');
    });

    it('SEC-013: Handles SQL injection in prompt', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const sqlButton = screen.getByTestId('sql-prompt-btn');
      await userEvent.click(sqlButton);
      
      // SQL injection should be passed through (sanitization is downstream)
      expect(mocks.mockSetInput).toHaveBeenCalledWith("'; DROP TABLE users; --");
    });

    it('SEC-014: Handles HTML img onerror injection', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const htmlButton = screen.getByTestId('html-prompt-btn');
      await userEvent.click(htmlButton);
      
      // HTML injection should be passed through
      expect(mocks.mockSetInput).toHaveBeenCalledWith('<img src=x onerror=alert(1)>');
    });
  });

  // ==========================================================================
  // CATEGORY 4: NAVIGATION ATTACKS
  // ==========================================================================
  describe('Navigation attack attempts', () => {
    it('SEC-015: Upload button navigates only to /documents (hardcoded)', async () => {
      vi.useRealTimers();
      
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      mocks.mockGetActiveVault.mockReturnValue({ name: 'Test Vault', file_count: 0 });
      
      renderWithRouter(['/chat/123']);
      
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      const uploadButton = screen.getByTestId('upload-btn');
      await userEvent.click(uploadButton);
      
      // Should navigate to /documents, not path traversal
      expect(mocks.mockNavigate).toHaveBeenCalledWith('/documents');
    });

    it('SEC-016: Session creation navigates to correct path', async () => {
      mocks.mockCreateChatSession.mockResolvedValue({ id: 999, title: 'New Chat' });
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Should navigate to /chat/999, not manipulated path
      expect(mocks.mockNavigate).toHaveBeenCalledWith('/chat/999', { replace: true });
    });

    it('SEC-017: URL hash fragments do not affect threadId parsing', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '123' }); // React Router decodes
      
      renderWithRouter(['/chat/123']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Thread ID parsed correctly
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(123);
    });

    it('SEC-018: Whitespace in threadId trimmed by parseInt', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '  42  ' });
      
      renderWithRouter(['/chat/  42  ']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt handles leading/trailing whitespace
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(42);
    });
  });

  // ==========================================================================
  // CATEGORY 5: BOUNDARY VIOLATIONS
  // ==========================================================================
  describe('Boundary violation tests', () => {
    it('SEC-019: Empty threadId triggers new session creation', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '' });
      
      renderWithRouter(['/chat/']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Empty string - parseInt returns NaN, so createChatSession should be called
      expect(mocks.mockCreateChatSession).toHaveBeenCalled();
    });

    it('SEC-020: Unicode numerals in threadId - not parsed as numbers', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '٤٢' }); // Arabic numerals
      
      renderWithRouter(['/chat/٤٢']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt('٤٢') returns NaN (not Western numerals)
      expect(mocks.mockLoadSession).not.toHaveBeenCalled();
    });

    it('SEC-021: Mixed content threadId - only leading number parsed', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '42<script>' });
      
      renderWithRouter(['/chat/42<script>']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt stops at non-numeric, returns 42
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(42);
    });

    it('SEC-022: Vault name with special chars displays safely', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      mocks.mockGetActiveVault.mockReturnValue({ name: 'Vault\x00Name', file_count: 5 });
      
      renderWithRouter(['/chat/123']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Component handles null bytes without crashing - displays with null byte visible
      expect(screen.getByText('Vault\x00Name')).toBeInTheDocument();
    });

    it('SEC-023: Undefined vault returns default name', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      mocks.mockGetActiveVault.mockReturnValue(undefined);
      
      renderWithRouter(['/chat/123']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Should use default vault name
      expect(screen.getByText('Default Vault')).toBeInTheDocument();
    });

    it('SEC-024: Very long vault name does not break layout', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '123' });
      mocks.mockGetActiveVault.mockReturnValue({ 
        name: 'A'.repeat(1000), 
        file_count: 5 
      });
      
      renderWithRouter(['/chat/123']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Should render without crashing
      expect(screen.getByText('A'.repeat(1000))).toBeInTheDocument();
    });
  });

  // ==========================================================================
  // CATEGORY 6: RACE CONDITIONS & CONCURRENCY
  // ==========================================================================
  describe('Race condition tests', () => {
    it('SEC-025: isCreatingSession guard prevents duplicate creation', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      mocks.mockCreateChatSession.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ id: 1, title: 'New Chat' }), 100))
      );
      
      renderWithRouter(['/chat']);
      
      // Attempt to create session
      await act(async () => {
        vi.advanceTimersByTime(10);
      });
      
      // The isCreatingSession guard should prevent duplicate calls
      // Verify at most one session creation attempt
      expect(mocks.mockCreateChatSession.mock.calls.length).toBeLessThanOrEqual(1);
    });

    it('SEC-026: Session creation error triggers error handler without crash', async () => {
      mocks.mockCreateChatSession.mockRejectedValue(new Error('Network error'));
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      // Should not throw - error is caught and logged
      expect(() => {
        renderWithRouter(['/chat']);
      }).not.toThrow();
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Error was logged but component didn't crash
      expect(mocks.mockCreateChatSession).toHaveBeenCalled();
    });
  });

  // ==========================================================================
  // CATEGORY 7: TYPE CONFUSION ATTACKS
  // ==========================================================================
  describe('Type confusion tests', () => {
    it('SEC-027: Array as threadId - toString conversion', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: ['1', '2', '3'] as any });
      
      renderWithRouter(['/chat/1,2,3']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // toString() called on array, results in "1,2,3" which parseInt converts to 1
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(1);
    });

    it('SEC-028: Object as threadId - [object Object] becomes NaN', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: { id: 42 } as any });
      
      renderWithRouter(['/chat/[object Object]']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt('[object Object]') = NaN, no session load
      expect(mocks.mockLoadSession).not.toHaveBeenCalled();
    });

    it('SEC-029: Boolean as threadId - becomes NaN', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: true as any });
      
      renderWithRouter(['/chat/true']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // parseInt('true') = NaN
      expect(mocks.mockLoadSession).not.toHaveBeenCalled();
    });
  });

  // ==========================================================================
  // CATEGORY 8: ERROR HANDLING
  // ==========================================================================
  describe('Error handling security', () => {
    it('SEC-030: Session load failure - error logged, no crash', async () => {
      mocks.mockLoadSession.mockRejectedValue(new Error('Session not found'));
      mocks.mockUseParams.mockReturnValue({ threadId: '999' });
      
      // Should not throw
      expect(() => {
        renderWithRouter(['/chat/999']);
      }).not.toThrow();
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Error was logged but component didn't crash
      expect(mocks.mockLoadSession).toHaveBeenCalled();
    });
  });
});
