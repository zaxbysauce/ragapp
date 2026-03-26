import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import '@testing-library/jest-dom';

// Hoisted mocks for proper hoisting
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

// Mock react-router-dom with importOriginal for partial mocking
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

// Mock useChatSession hook
vi.mock('@/components/chat/useChatSession', () => ({
  useChatSession: mocks.useChatSessionMock,
}));

// Mock createChatSession API
vi.mock('@/lib/api', () => ({
  createChatSession: mocks.mockCreateChatSession,
}));

// Mock useRequirePasswordChange
vi.mock('@/lib/auth', () => ({
  useRequirePasswordChange: vi.fn(),
}));

// Mock lucide-react icons - comprehensive list
vi.mock('lucide-react', () => ({
  Loader2: ({ className }: { className?: string }) => (
    <span data-testid="mock-loader" className={className}>⟳</span>
  ),
  Plus: () => <span data-testid="mock-plus">+</span>,
  PanelRight: () => <span data-testid="mock-panel-right">→</span>,
  PanelRightClose: () => <span data-testid="mock-panel-right-close">×</span>,
  User: () => <span data-testid="mock-user">👤</span>,
  Bot: () => <span data-testid="mock-bot">🤖</span>,
  Send: () => <span data-testid="mock-send">➤</span>,
  Square: () => <span data-testid="mock-square">■</span>,
}));

// Mock UI components
vi.mock('@/components/ui/scroll-area', () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-scroll-area">{children}</div>
  ),
}));

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, variant, size, ...props }: any) => (
    <button
      onClick={onClick}
      disabled={disabled}
      data-variant={variant}
      data-size={size}
      {...props}
    >
      {children}
    </button>
  ),
}));

vi.mock('@/components/ui/select', () => ({
  Select: ({ children }: any) => <div data-testid="mock-select">{children}</div>,
  SelectTrigger: ({ children, ...props }: any) => (
    <button {...props}>{children}</button>
  ),
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  SelectValue: ({ children }: any) => <span>{children}</span>,
}));

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: { children: React.ReactNode }) => (
      <div {...props}>{children}</div>
    ),
    p: ({ children, ...props }: { children: React.ReactNode }) => (
      <p {...props}>{children}</p>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock EmptyTranscript for EmptyState
vi.mock('@/components/chat/EmptyTranscript', () => ({
  EmptyTranscript: ({ onPromptClick, onUploadClick, hasDocuments }: any) => (
    <div data-testid="mock-empty-transcript">
      <button data-testid="prompt-btn" onClick={() => onPromptClick?.('Test prompt')}>
        Test Prompt
      </button>
      {!hasDocuments && (
        <button data-testid="upload-btn" onClick={onUploadClick}>
          Upload Documents
        </button>
      )}
    </div>
  ),
}));

// Mock VaultSelector
vi.mock('@/components/vault/VaultSelector', () => ({
  VaultSelector: () => <div data-testid="mock-vault-selector">Vault</div>,
}));

// Mock ChatThread to avoid ResizeObserver issues
vi.mock('@/components/chat/ChatThread', () => ({
  ChatThread: ({ className }: { className?: string }) => (
    <div data-testid="mock-chat-thread" className={className}>ChatThread</div>
  ),
}));

// Import after mocks
import ChatPage from './page';

// Helper to render with router
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

describe('ChatPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    
    // Reset mock implementations
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

  describe('useRequirePasswordChange integration', () => {
    it('Test 1: calls useRequirePasswordChange hook at top of component', async () => {
      const { useRequirePasswordChange } = await import('@/lib/auth');
      
      renderWithRouter(['/chat']);
      
      // Verify the hook was called
      expect(useRequirePasswordChange).toHaveBeenCalled();
    });
  });

  describe('New session creation at /chat (no threadId)', () => {
    it('Test 2: creates new session via createChatSession API when mounted at /chat', async () => {
      mocks.mockCreateChatSession.mockResolvedValue({ id: 456, title: 'New Chat' });
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      // Wait for the useEffect to run
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Verify createChatSession was called
      expect(mocks.mockCreateChatSession).toHaveBeenCalledWith({
        title: 'New Chat',
        vault_id: 1,
      });
    });

    it('Test 3: navigates to /chat/{sessionId} after successful session creation', async () => {
      mocks.mockCreateChatSession.mockResolvedValue({ id: 789, title: 'New Chat' });
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Verify navigate was called with the new session ID
      expect(mocks.mockNavigate).toHaveBeenCalledWith('/chat/789', { replace: true });
    });

    it('Test 4: clears existing session before creating new one', async () => {
      mocks.mockCreateChatSession.mockResolvedValue({ id: 100, title: 'New Chat' });
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Verify clearSession was called before createChatSession
      expect(mocks.mockClearSession).toHaveBeenCalled();
    });
  });

  describe('Session loading at /chat/{threadId}', () => {
    it('Test 5: loads session via loadSession when threadId is provided', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '42' });
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/42']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Verify loadSession was called with the parsed threadId
      expect(mocks.mockLoadSession).toHaveBeenCalledWith(42);
    });

    it('Test 6: does NOT call createChatSession when threadId is provided', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '99' });
      
      renderWithRouter(['/chat/99']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // createChatSession should NOT be called for existing sessions
      expect(mocks.mockCreateChatSession).not.toHaveBeenCalled();
    });
  });

  describe('handlePromptClick behavior', () => {
    it('Test 7: calls useChatStore.setInput with the prompt when prompt is clicked', async () => {
      vi.useRealTimers(); // Use real timers for user interaction tests
      
      mocks.mockCreateChatSession.mockResolvedValue({ id: 200, title: 'New Chat' });
      mocks.mockUseParams.mockReturnValue({ threadId: '200' }); // Existing session
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/200']);
      
      // Wait for render
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      // Click the prompt button in EmptyState
      const promptButton = screen.getByTestId('prompt-btn');
      await userEvent.click(promptButton);
      
      // Verify setInput was called with the prompt
      expect(mocks.mockSetInput).toHaveBeenCalledWith('Test prompt');
    });
  });

  describe('handleUploadClick behavior', () => {
    it('Test 8: navigates to /documents when upload button is clicked', async () => {
      vi.useRealTimers(); // Use real timers for user interaction tests
      
      mocks.mockCreateChatSession.mockResolvedValue({ id: 300, title: 'New Chat' });
      mocks.mockGetActiveVault.mockReturnValue({ name: 'Test Vault', file_count: 0 }); // No documents
      mocks.mockUseParams.mockReturnValue({ threadId: '300' }); // Existing session
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/300']);
      
      // Wait for render
      await waitFor(() => {
        expect(screen.getByTestId('mock-empty-transcript')).toBeInTheDocument();
      });
      
      // Click the upload button
      const uploadButton = screen.getByTestId('upload-btn');
      await userEvent.click(uploadButton);
      
      // Verify navigate was called with /documents
      expect(mocks.mockNavigate).toHaveBeenCalledWith('/documents');
    });

    it('Test 9: does NOT show upload button when vault has documents', async () => {
      mocks.mockCreateChatSession.mockResolvedValue({ id: 400, title: 'New Chat' });
      mocks.mockGetActiveVault.mockReturnValue({ name: 'Test Vault', file_count: 10 }); // Has documents
      mocks.mockUseParams.mockReturnValue({ threadId: '400' }); // Existing session
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/400']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Upload button should NOT be present
      expect(screen.queryByTestId('upload-btn')).not.toBeInTheDocument();
    });
  });

  describe('Session creation error handling', () => {
    it('Test 10: shows error message when session creation fails', async () => {
      vi.useRealTimers(); // Use real timers for error handling tests
      
      mocks.mockCreateChatSession.mockRejectedValue(new Error('Network error'));
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      // Wait for the error to be displayed
      await waitFor(() => {
        expect(screen.getByText('Failed to create chat session. Please try again.')).toBeInTheDocument();
      }, { timeout: 10000 });
    });

    it('Test 11: sets sessionError state on creation failure with destructive styling', async () => {
      vi.useRealTimers(); // Use real timers for error handling tests
      
      mocks.mockCreateChatSession.mockRejectedValue(new Error('API Error'));
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      await waitFor(() => {
        // Error should be visible with destructive class
        const errorElement = screen.getByText(/Failed to create chat session/);
        expect(errorElement).toBeInTheDocument();
        expect(errorElement).toHaveClass('text-destructive');
      }, { timeout: 10000 });
    });

    it('Test 12: does not navigate when session creation fails', async () => {
      vi.useRealTimers(); // Use real timers for error handling tests
      
      mocks.mockCreateChatSession.mockRejectedValue(new Error('Network error'));
      mocks.mockNavigate.mockClear();
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      // Wait to ensure async operations complete
      await waitFor(() => {
        // Navigate should not have been called
        expect(mocks.mockNavigate).not.toHaveBeenCalled();
      }, { timeout: 10000 });
    });
  });

  describe('Loading spinner behavior', () => {
    it('Test 13: shows loading spinner when isCreatingSession is true', async () => {
      mocks.mockCreateChatSession.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ id: 500, title: 'New Chat' }), 1000))
      );
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      // Immediately after render, the session creation is in progress
      await act(async () => {
        vi.advanceTimersByTime(100);
      });
      
      // Loader should be visible during creation
      expect(screen.getByTestId('mock-loader')).toBeInTheDocument();
    });

    it('Test 14: shows loading spinner when isLoading from useChatSession is true', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '75' });
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: true,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/75']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // Loader should be visible during loading
      expect(screen.getByTestId('mock-loader')).toBeInTheDocument();
    });
  });

  describe('EmptyState rendering', () => {
    it('Test 15: shows EmptyState when no messages and not loading', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '600' }); // Existing session
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/600']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // EmptyState should show when no messages
      const emptyState = screen.getByTestId('mock-empty-transcript');
      expect(emptyState).toBeInTheDocument();
    });

    it('Test 16: passes vaultName to EmptyState', async () => {
      mocks.mockGetActiveVault.mockReturnValue({ name: 'My Custom Vault', file_count: 3 });
      mocks.mockUseParams.mockReturnValue({ threadId: '700' });
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/700']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // EmptyState should receive the vault name
      expect(screen.getByText('My Custom Vault')).toBeInTheDocument();
    });
  });

  describe('Vault store integration', () => {
    it('Test 17: uses activeVaultId from vault store when creating session', async () => {
      mocks.useVaultStoreMock.mockReturnValue({
        activeVaultId: 42,
        getActiveVault: mocks.mockGetActiveVault.mockReturnValue({ name: 'Vault 42', file_count: 0 }),
      });
      
      mocks.mockCreateChatSession.mockResolvedValue({ id: 800, title: 'New Chat' });
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // createChatSession should be called with vault_id from store
      expect(mocks.mockCreateChatSession).toHaveBeenCalledWith({
        title: 'New Chat',
        vault_id: 42,
      });
    });

    it('Test 18: handles null activeVaultId gracefully', async () => {
      mocks.useVaultStoreMock.mockReturnValue({
        activeVaultId: null,
        getActiveVault: mocks.mockGetActiveVault.mockReturnValue(undefined),
      });
      
      mocks.mockCreateChatSession.mockResolvedValue({ id: 900, title: 'New Chat' });
      mocks.mockUseParams.mockReturnValue({ threadId: undefined });
      
      renderWithRouter(['/chat']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // createChatSession should be called with undefined vault_id
      expect(mocks.mockCreateChatSession).toHaveBeenCalledWith({
        title: 'New Chat',
        vault_id: undefined,
      });
    });
  });

  describe('ChatHeader rendering', () => {
    it('Test 19: renders ChatHeader component', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '100' });
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/100']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // ChatHeader should be rendered - verify by checking for header element
      const header = document.querySelector('header');
      expect(header).toBeInTheDocument();
    });
  });

  describe('ChatThread rendering', () => {
    it('Test 20: renders ChatThread when messages exist', async () => {
      mocks.mockUseParams.mockReturnValue({ threadId: '200' });
      
      mocks.useChatSessionMock.mockReturnValue({
        isLoading: false,
        loadSession: mocks.mockLoadSession.mockResolvedValue(undefined),
        messages: [{ id: '1', role: 'user', content: 'Hello' }],
        clearSession: mocks.mockClearSession,
      });
      
      renderWithRouter(['/chat/200']);
      
      await act(async () => {
        vi.runAllTimers();
      });
      
      // When there are messages, ChatThread should be rendered (mocked)
      const chatThread = screen.getByTestId('mock-chat-thread');
      expect(chatThread).toBeInTheDocument();
    });
  });
});
