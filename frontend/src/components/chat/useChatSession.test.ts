import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act } from 'react';

// Hoist mocks to top level
const { mockGet, mockPost } = vi.hoisted(() => {
  return {
    mockGet: vi.fn(),
    mockPost: vi.fn(),
  };
});

// Mock react-router-dom BEFORE importing useChatSession
vi.mock('react-router-dom', () => ({
  useParams: vi.fn().mockReturnValue({ sessionId: '1' }),
}));

// Mock @/lib/api with default export
vi.mock('@/lib/api', () => {
  const mockApiClient = {
    get: mockGet,
    post: mockPost,
  };
  return {
    default: mockApiClient,
    apiClient: mockApiClient,
  };
});

// Now import the store
import { useChatSessionStore, type ChatSessionDetail, type ChatSessionMessage } from './useChatSession';

describe('useChatSessionStore', () => {
  const mockSession: ChatSessionDetail = {
    id: 1,
    title: 'Test Session',
    messages: [],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  const mockMessage: ChatSessionMessage = {
    id: '1',
    role: 'assistant',
    content: 'Hello!',
    sources: [],
    created_at: new Date().toISOString(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store state
    useChatSessionStore.setState({
      session: null,
      messages: [],
      isLoading: false,
      isStreaming: false,
      streamingContent: '',
    });
  });

  describe('loadSession', () => {
    it('should load session and reset streaming state', async () => {
      mockGet.mockResolvedValue({ data: { ...mockSession, messages: [mockMessage] } });

      await act(async () => {
        await useChatSessionStore.getState().loadSession(1);
      });

      const state = useChatSessionStore.getState();
      // Check session properties individually to avoid timestamp issues
      expect(state.session?.id).toBe(1);
      expect(state.session?.title).toBe('Test Session');
      expect(state.messages).toEqual([mockMessage]);
      expect(state.isLoading).toBe(false);
      expect(state.isStreaming).toBe(false);
      expect(state.streamingContent).toBe('');
    });

    it('should reset streamingContent to empty string on load', async () => {
      // Set some streaming content first
      useChatSessionStore.setState({
        streamingContent: 'Partial response...',
        isStreaming: true,
      });

      mockGet.mockResolvedValue({ data: mockSession });

      await act(async () => {
        await useChatSessionStore.getState().loadSession(1);
      });

      expect(useChatSessionStore.getState().streamingContent).toBe('');
      expect(useChatSessionStore.getState().isStreaming).toBe(false);
    });

    it('should throw error when API fails', async () => {
      mockGet.mockRejectedValue(new Error('Failed to load'));

      await expect(
        act(async () => {
          await useChatSessionStore.getState().loadSession(1);
        })
      ).rejects.toThrow('Failed to load');
    });
  });

  describe('onStreamChunk', () => {
    it('should accumulate streaming content and set isStreaming true', () => {
      act(() => {
        useChatSessionStore.getState().onStreamChunk('Hello ');
      });

      expect(useChatSessionStore.getState().streamingContent).toBe('Hello ');
      expect(useChatSessionStore.getState().isStreaming).toBe(true);

      act(() => {
        useChatSessionStore.getState().onStreamChunk('world!');
      });

      expect(useChatSessionStore.getState().streamingContent).toBe('Hello world!');
    });
  });

  describe('onStreamDone', () => {
    beforeEach(() => {
      // Set up session for onStreamDone
      useChatSessionStore.setState({ session: mockSession });
      mockPost.mockResolvedValue({ data: mockMessage });
    });

    it('should save message and reset streaming state on success', async () => {
      // Set streaming content
      useChatSessionStore.setState({
        streamingContent: 'Complete response',
        isStreaming: true,
      });

      await act(async () => {
        await useChatSessionStore.getState().onStreamDone('Complete response', []);
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/chat/sessions/1/messages',
        { role: 'assistant', content: 'Complete response', sources: [] }
      );
      expect(useChatSessionStore.getState().isStreaming).toBe(false);
      expect(useChatSessionStore.getState().streamingContent).toBe('');
    });

    it('should reset streaming state even when no session', async () => {
      useChatSessionStore.setState({ session: null });

      await act(async () => {
        await useChatSessionStore.getState().onStreamDone('Response', []);
      });

      expect(useChatSessionStore.getState().isStreaming).toBe(false);
      expect(useChatSessionStore.getState().streamingContent).toBe('');
    });

    it('should add saved message to messages array', async () => {
      useChatSessionStore.setState({ messages: [] });

      await act(async () => {
        await useChatSessionStore.getState().onStreamDone('Response', []);
      });

      const messages = useChatSessionStore.getState().messages;
      expect(messages.length).toBe(1);
      expect(messages[0]).toEqual(mockMessage);
    });
  });

  describe('clearSession', () => {
    it('should reset all state to initial values', () => {
      // Set up some state
      useChatSessionStore.setState({
        session: mockSession,
        messages: [mockMessage],
        isLoading: true,
        isStreaming: true,
        streamingContent: 'Some content',
      });

      act(() => {
        useChatSessionStore.getState().clearSession();
      });

      const state = useChatSessionStore.getState();
      expect(state.session).toBeNull();
      expect(state.messages).toEqual([]);
      expect(state.isLoading).toBe(false);
      expect(state.isStreaming).toBe(false);
      expect(state.streamingContent).toBe('');
    });
  });

  describe('CRITICAL: streamingContent cleanup on error (Task 4.6 fix)', () => {
    /**
     * These tests verify the fix for Task 4.6
     * 
     * Bug: When onStreamDone caught an error, it only set isStreaming: false
     * but left streamingContent with the partial content.
     * 
     * Fix: Changed catch block to also set streamingContent: ""
     */

    it('must clear streamingContent when saveMessage fails', async () => {
      // Set up session and some streaming content
      useChatSessionStore.setState({
        session: mockSession,
        streamingContent: 'This is a partial response that was streaming',
        isStreaming: true,
      });

      mockPost.mockRejectedValue(new Error('Network error'));

      let thrownError: Error | undefined;
      
      // Use a try/catch pattern within act to properly handle the rejection
      await act(async () => {
        try {
          await useChatSessionStore.getState().onStreamDone('Partial', []);
        } catch (e) {
          thrownError = e as Error;
        }
      });

      // CRITICAL ASSERTION: streamingContent must be empty after error
      const state = useChatSessionStore.getState();
      expect(thrownError?.message).toBe('Network error');
      expect(state.isStreaming).toBe(false);
      expect(state.streamingContent).toBe(''); // This was the bug fix!
    });

    it('must clear streamingContent when no session exists', async () => {
      useChatSessionStore.setState({
        session: null,
        streamingContent: 'Content during session-less stream',
        isStreaming: true,
      });

      await act(async () => {
        await useChatSessionStore.getState().onStreamDone('Some content', []);
      });

      const state = useChatSessionStore.getState();
      expect(state.streamingContent).toBe('');
      expect(state.isStreaming).toBe(false);
    });

    it('must clear both isStreaming and streamingContent on error path', async () => {
      useChatSessionStore.setState({
        session: mockSession,
        streamingContent: 'Partial content',
        isStreaming: true,
      });

      mockPost.mockRejectedValue(new Error('API Error'));

      await act(async () => {
        try {
          await useChatSessionStore.getState().onStreamDone('Content', []);
        } catch {
          // Expected
        }
      });

      const state = useChatSessionStore.getState();
      // Both must be reset
      expect(state.isStreaming).toBe(false);
      expect(state.streamingContent).toBe('');
    });
  });
});
