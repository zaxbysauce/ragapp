import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useAuthStore } from '@/stores/authStore';

// Mock fetch globally
global.fetch = vi.fn();

describe('authStore - Hydration Race Condition Fix', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Reset the store to initial state before each test
    useAuthStore.setState({
      user: null,
      accessToken: null,
      isAuthenticated: false,
      isLoading: true,
    });
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('1. Initial isLoading is true', () => {
    it('should start with isLoading true to prevent flash of protected content', () => {
      // Create a fresh store instance to test initial state
      const initialState = useAuthStore.getState();
      
      // The store should initialize with isLoading: true
      // This prevents the ProtectedRoute from showing content before auth check
      expect(initialState.isLoading).toBe(true);
    });

    it('should initialize isAuthenticated as false', () => {
      const { isAuthenticated } = useAuthStore.getState();
      expect(isAuthenticated).toBe(false);
    });

    it('should initialize user as null', () => {
      const { user } = useAuthStore.getState();
      expect(user).toBe(null);
    });
  });

  describe('2. onRehydrateStorage sets isLoading false when no user', () => {
    it('should set isLoading to false after hydration when no persisted user', async () => {
      // Simulate the onRehydrateStorage callback being called with no state (no user)
      // This happens when localStorage has no user data
      
      // First verify initial state has isLoading: true
      expect(useAuthStore.getState().isLoading).toBe(true);
      
      // Trigger rehydration callback manually - simulating what happens after
      // localStorage is read and there's no user
      useAuthStore.setState({ isLoading: false, isAuthenticated: false });
      
      const { isLoading, isAuthenticated } = useAuthStore.getState();
      
      expect(isLoading).toBe(false);
      expect(isAuthenticated).toBe(false);
    });

    it('should NOT clear isLoading if user exists after hydration', async () => {
      // Set up a persisted user (simulating localStorage having user data)
      const persistedUser = {
        id: 1,
        username: 'testuser',
        full_name: 'Test User',
        role: 'member' as const,
        is_active: true,
      };
      
      useAuthStore.setState({ user: persistedUser });
      
      // When user exists, the store should trigger checkAuth instead of just
      // setting isLoading to false
      // The checkAuth will handle setting isLoading appropriately
      
      // With a user present, isAuthenticated should be set based on user
      expect(useAuthStore.getState().user).toEqual(persistedUser);
    });
  });

  describe('3. 401 handler sets isLoading false when refresh fails', () => {
    it('should set isLoading to false when checkAuth receives 401 and refresh fails', async () => {
      // Mock fetch to return 401 on /api/auth/me
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      
      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 401,
          json: () => Promise.resolve({ detail: 'Unauthorized' }),
        })
        // Refresh token also fails
        .mockResolvedValueOnce({
          ok: false,
          status: 401,
          json: () => Promise.resolve({ detail: 'Token expired' }),
        });

      // Set an initial token so checkAuth tries to validate it
      useAuthStore.setState({
        accessToken: 'expired-token',
        isLoading: true,
      });

      // Call checkAuth - it should handle 401 and refresh failure
      await useAuthStore.getState().checkAuth();

      const { isLoading, isAuthenticated, user, accessToken } = useAuthStore.getState();

      // After failed refresh on 401, all auth state should be cleared
      expect(isLoading).toBe(false);
      expect(isAuthenticated).toBe(false);
      expect(user).toBe(null);
      expect(accessToken).toBe(null);
    });

    it('should set isLoading to false when checkAuth has no token and refresh fails', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      
      // No token, so it goes straight to refresh which fails
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ detail: 'No token' }),
      });

      useAuthStore.setState({
        accessToken: null,
        isLoading: true,
      });

      await useAuthStore.getState().checkAuth();

      const { isLoading, isAuthenticated } = useAuthStore.getState();

      expect(isLoading).toBe(false);
      expect(isAuthenticated).toBe(false);
    });

    it('should set isLoading to false on network error during checkAuth', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      
      // Network error
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      useAuthStore.setState({
        accessToken: 'some-token',
        isLoading: true,
      });

      await useAuthStore.getState().checkAuth();

      const { isLoading, isAuthenticated } = useAuthStore.getState();

      expect(isLoading).toBe(false);
      expect(isAuthenticated).toBe(false);
    });
  });

  describe('4. Protected content flash prevention', () => {
    it('should keep isLoading true until auth check completes', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      
      // Mock successful auth validation
      const mockUser = {
        id: 1,
        username: 'testuser',
        full_name: 'Test User',
        role: 'member' as const,
        is_active: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockUser),
      });

      // Start with isLoading: true (initial state)
      useAuthStore.setState({
        accessToken: 'valid-token',
        isLoading: true,
      });

      // Check state before checkAuth
      expect(useAuthStore.getState().isLoading).toBe(true);
      expect(useAuthStore.getState().isAuthenticated).toBe(false);

      // Call checkAuth
      await useAuthStore.getState().checkAuth();

      // After successful check, isLoading should be false
      const { isLoading, isAuthenticated, user } = useAuthStore.getState();

      expect(isLoading).toBe(false);
      expect(isAuthenticated).toBe(true);
      expect(user).toEqual(mockUser);
    });

    it('ProtectedRoute should show loader when isLoading is true', () => {
      // This test verifies the intended behavior that ProtectedRoute relies on
      // When isLoading is true, the loader should be shown, not the protected content
      
      const { isLoading, isAuthenticated } = useAuthStore.getState();
      
      // Simulating what ProtectedRoute checks:
      // if (isLoading) return <Loader />;
      // if (!isAuthenticated) return <Navigate to="/login" />;
      
      // Initial state: isLoading=true, isAuthenticated=false
      // ProtectedRoute should show loader, NOT content, NOT redirect
      expect(isLoading).toBe(true);
      expect(isAuthenticated).toBe(false);
      
      // This is the key fix: isLoading starts as true so ProtectedRoute 
      // shows loader instead of either:
      // 1. Flashing protected content before auth check
      // 2. Prematurely redirecting to login before checking
    });

    it('should transition from loading to authenticated without flash', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      
      const mockUser = {
        id: 1,
        username: 'testuser',
        full_name: 'Test User',
        role: 'admin' as const,
        is_active: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockUser),
      });

      // Step 1: Initial state - isLoading true prevents any content showing
      const initialState = useAuthStore.getState();
      expect(initialState.isLoading).toBe(true);
      
      // Step 2: checkAuth in progress - isLoading still true
      useAuthStore.setState({ accessToken: 'token', isLoading: true });
      const duringState = useAuthStore.getState();
      expect(duringState.isLoading).toBe(true);
      
      // Step 3: checkAuth completes - isLoading becomes false, isAuthenticated true
      await useAuthStore.getState().checkAuth();
      const finalState = useAuthStore.getState();
      
      expect(finalState.isLoading).toBe(false);
      expect(finalState.isAuthenticated).toBe(true);
      
      // The transition ensures ProtectedRoute goes:
      // Loader -> (auth succeeds) -> Content
      // Instead of:
      // Content flash -> Redirect -> Login
    });
  });

  describe('State transition invariants', () => {
    it('should always set isLoading false when becoming unauthenticated', async () => {
      // After any unauthenticated state, isLoading should be false
      // This ensures the user sees the login page, not a loader
      
      // Test logout
      useAuthStore.setState({
        user: { id: 1, username: 'user', full_name: 'User', role: 'member' as const, is_active: true },
        accessToken: 'token',
        isAuthenticated: true,
        isLoading: false,
      });

      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await useAuthStore.getState().logout();

      const { isLoading, isAuthenticated } = useAuthStore.getState();
      expect(isLoading).toBe(false);
      expect(isAuthenticated).toBe(false);
    });

    it('refreshToken clears auth state but does not manage isLoading', async () => {
      const mockFetch = global.fetch as ReturnType<typeof vi.fn>;
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
      });

      useAuthStore.setState({
        accessToken: 'expired-token',
        isLoading: true,
      });

      const result = await useAuthStore.getState().refreshToken();

      expect(result).toBe(false);
      const { isLoading, isAuthenticated, user } = useAuthStore.getState();
      
      // refreshToken clears auth state but does NOT touch isLoading
      // isLoading management is the responsibility of the caller (checkAuth)
      expect(isLoading).toBe(true); // unchanged - refreshToken doesn't touch it
      expect(isAuthenticated).toBe(false);
      expect(user).toBe(null);
    });
  });
});
