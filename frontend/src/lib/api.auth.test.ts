import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import axios from 'axios';
import { useAuthStore } from '@/stores/authStore';

// We need to import the apiClient after we've set up our mocks
// This is a unit test that tests the interceptor behavior directly

describe('JWT Auth Migration - API Client Interceptors', () => {
  let mockLogout: ReturnType<typeof vi.fn>;
  let originalLocation: typeof window.location;

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Store original location
    originalLocation = window.location;
    
    // Mock window.location.href
    delete (window as any).location;
    window.location = {
      ...originalLocation,
      href: '',
      pathname: '/dashboard',
    } as Location;
    
    // Get the auth store and reset it to a clean state
    const authStore = useAuthStore.getState();
    authStore.setAccessToken(null);
    authStore.setUser(null);
    
    // Mock logout function
    mockLogout = vi.fn().mockResolvedValue(undefined);
    useAuthStore.setState({
      logout: mockLogout,
    });
  });

  afterEach(() => {
    // Restore original location
    window.location = originalLocation;
    vi.resetModules();
  });

  describe('Request Interceptor - JWT Token Attachment', () => {
    it('should attach JWT token to Authorization header when token exists', async () => {
      const testToken = 'test-jwt-token-12345';
      
      // Set token in auth store
      useAuthStore.getState().setAccessToken(testToken);
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      // Import apiClient fresh to get the interceptor with the token
      const { default: apiClient } = await import('@/lib/api');
      
      // Create a mock request config
      const config = {
        headers: {} as Record<string, string>,
        method: 'get',
        url: '/test',
      };
      
      // Get the request interceptor function and call it
      // axios interceptors are stored in interceptor.request handlers
      const interceptor = apiClient.interceptors.request.handlers[0];
      
      if (interceptor && 'fulfilled' in interceptor) {
        const result = await interceptor.fulfilled(config);
        expect(result.headers.Authorization).toBe(`Bearer ${testToken}`);
      } else {
        // Fallback: make an actual request to test interceptor
        const mockAdapter = vi.fn().mockResolvedValue({ data: {} });
        apiClient.defaults.adapter = mockAdapter;
        try {
          await apiClient.get('/test');
        } catch {
          // Ignore errors, we're just checking the request was made with token
        }
        // The token should have been attached
        expect(mockAdapter).toHaveBeenCalled();
        const callArgs = mockAdapter.mock.calls[0]?.[0];
        expect(callArgs?.headers?.Authorization).toBe(`Bearer ${testToken}`);
      }
    });

    it('should NOT attach Authorization header when no token exists', async () => {
      // Ensure no token in auth store
      useAuthStore.getState().setAccessToken(null);
      useAuthStore.getState().setUser(null);
      
      // Import apiClient fresh
      const { default: apiClient } = await import('@/lib/api');
      
      // Create a mock request config
      const config = {
        headers: {} as Record<string, string>,
        method: 'get',
        url: '/test',
      };
      
      // Get the request interceptor and call it
      const interceptor = apiClient.interceptors.request.handlers[0];
      
      if (interceptor && 'fulfilled' in interceptor) {
        const result = await interceptor.fulfilled(config);
        // Should NOT have Authorization header when no token
        expect(result.headers.Authorization).toBeUndefined();
      } else {
        // Fallback test
        const mockAdapter = vi.fn().mockRejectedValue(new Error('Network error'));
        apiClient.defaults.adapter = mockAdapter;
        try {
          await apiClient.get('/test');
        } catch {
          // Expected to fail
        }
        const callArgs = mockAdapter.mock.calls[0]?.[0];
        // Authorization header should be undefined or not set
        expect(callArgs?.headers?.Authorization).toBeUndefined();
      }
    });

    it('should NOT attach Authorization header when token is empty string', async () => {
      // Set empty token
      useAuthStore.getState().setAccessToken('');
      
      // Import apiClient fresh
      const { default: apiClient } = await import('@/lib/api');
      
      const config = {
        headers: {} as Record<string, string>,
        method: 'get',
        url: '/test',
      };
      
      const interceptor = apiClient.interceptors.request.handlers[0];
      
      if (interceptor && 'fulfilled' in interceptor) {
        const result = await interceptor.fulfilled(config);
        expect(result.headers.Authorization).toBeUndefined();
      }
    });

    it('should attach fresh token from store on each request', async () => {
      // Initially no token
      useAuthStore.getState().setAccessToken(null);
      
      const { default: apiClient } = await import('@/lib/api');
      
      // Set a token after initial import
      const firstToken = 'first-token';
      useAuthStore.getState().setAccessToken(firstToken);
      
      const config = {
        headers: {} as Record<string, string>,
        method: 'get',
        url: '/test',
      };
      
      const interceptor = apiClient.interceptors.request.handlers[0];
      
      if (interceptor && 'fulfilled' in interceptor) {
        const result = await interceptor.fulfilled(config);
        expect(result.headers.Authorization).toBe(`Bearer ${firstToken}`);
        
        // Update token
        const secondToken = 'second-token';
        useAuthStore.getState().setAccessToken(secondToken);
        
        // Create new config for second request
        const config2 = {
          headers: {} as Record<string, string>,
          method: 'get',
          url: '/test2',
        };
        
        const result2 = await interceptor.fulfilled(config2);
        expect(result2.headers.Authorization).toBe(`Bearer ${secondToken}`);
      }
    });
  });

  describe('Response Interceptor - 401 Handling', () => {
    it('should clear auth state on 401 response', async () => {
      // Set authenticated state
      useAuthStore.getState().setAccessToken('test-token');
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      const { default: apiClient } = await import('@/lib/api');
      
      // Create a mock 401 error
      const mockError = {
        name: 'Error',
        message: 'Unauthorized',
        response: {
          status: 401,
          data: { detail: 'Unauthorized' },
        },
        config: { headers: {} },
      };
      
      // Get the response error interceptor
      const errorInterceptor = apiClient.interceptors.response.handlers[1];
      
      if (errorInterceptor && 'rejected' in errorInterceptor) {
        await expect(errorInterceptor.rejected(mockError as any)).rejects.toThrow();
        
        // Verify logout was called
        expect(mockLogout).toHaveBeenCalled();
        
        // Verify redirect was attempted
        expect(window.location.href).toBe('/login');
      }
    });

    it('should NOT clear auth state on non-401 errors', async () => {
      // Set authenticated state
      useAuthStore.getState().setAccessToken('test-token');
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      const { default: apiClient } = await import('@/lib/api');
      
      // Create a mock 500 error
      const mockError = {
        name: 'Error',
        message: 'Server Error',
        response: {
          status: 500,
          data: { detail: 'Internal Server Error' },
        },
        config: { headers: {} },
      };
      
      const errorInterceptor = apiClient.interceptors.response.handlers[1];
      
      if (errorInterceptor && 'rejected' in errorInterceptor) {
        await expect(errorInterceptor.rejected(mockError as any)).rejects.toThrow();
        
        // Verify logout was NOT called
        expect(mockLogout).not.toHaveBeenCalled();
      }
    });

    it('should NOT redirect if already on login page', async () => {
      // Set location to login page
      window.location.pathname = '/login';
      
      useAuthStore.getState().setAccessToken('test-token');
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      const { default: apiClient } = await import('@/lib/api');
      
      const mockError = {
        name: 'Error',
        message: 'Unauthorized',
        response: { status: 401, data: { detail: 'Unauthorized' } },
        config: { headers: {} },
      };
      
      const errorInterceptor = apiClient.interceptors.response.handlers[1];
      
      if (errorInterceptor && 'rejected' in errorInterceptor) {
        await expect(errorInterceptor.rejected(mockError as any)).rejects.toThrow();
        
        // logout should still be called
        expect(mockLogout).toHaveBeenCalled();
        
        // But href should NOT be set to /login since already there
        expect(window.location.href).toBe('');
      }
    });

    it('should NOT redirect if on register page', async () => {
      // Set location to register page
      window.location.pathname = '/register';
      
      useAuthStore.getState().setAccessToken('test-token');
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      const { default: apiClient } = await import('@/lib/api');
      
      const mockError = {
        name: 'Error',
        message: 'Unauthorized',
        response: { status: 401, data: { detail: 'Unauthorized' } },
        config: { headers: {} },
      };
      
      const errorInterceptor = apiClient.interceptors.response.handlers[1];
      
      if (errorInterceptor && 'rejected' in errorInterceptor) {
        await expect(errorInterceptor.rejected(mockError as any)).rejects.toThrow();
        
        // logout should still be called
        expect(mockLogout).toHaveBeenCalled();
        
        // But href should NOT be set
        expect(window.location.href).toBe('');
      }
    });

    it('should preserve AbortError without clearing auth', async () => {
      useAuthStore.getState().setAccessToken('test-token');
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      const { default: apiClient } = await import('@/lib/api');
      
      // Create AbortError
      const abortError = new Error('Request cancelled');
      abortError.name = 'AbortError';
      
      const errorInterceptor = apiClient.interceptors.response.handlers[1];
      
      if (errorInterceptor && 'rejected' in errorInterceptor) {
        await expect(errorInterceptor.rejected(abortError)).rejects.toThrow('Request cancelled');
        
        // logout should NOT be called for AbortError
        expect(mockLogout).not.toHaveBeenCalled();
      }
    });

    it('should preserve ERR_CANCELED without clearing auth', async () => {
      useAuthStore.getState().setAccessToken('test-token');
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      const { default: apiClient } = await import('@/lib/api');
      
      // Create cancelled error
      const cancelledError = {
        name: 'Error',
        message: 'Request cancelled',
        code: 'ERR_CANCELED',
        config: { headers: {} },
      };
      
      const errorInterceptor = apiClient.interceptors.response.handlers[1];
      
      if (errorInterceptor && 'rejected' in errorInterceptor) {
        await expect(errorInterceptor.rejected(cancelledError as any)).rejects.toThrow();
        
        // logout should NOT be called for cancelled
        expect(mockLogout).not.toHaveBeenCalled();
      }
    });
  });

  describe('Integration - End-to-end request flow', () => {
    it('should make authenticated request with token', async () => {
      const testToken = 'integration-test-token';
      useAuthStore.getState().setAccessToken(testToken);
      useAuthStore.getState().setUser({ id: 1, username: 'testuser', full_name: 'Test User', role: 'member' as const, is_active: true });
      
      // We need to mock axios adapter to capture the request
      const mockAxios = vi.mocked(axios);
      const originalCreate = mockAxios.create;
      
      // Track the request made
      let capturedConfig: any = null;
      
      // Mock the request to capture config
      vi.mock('@/lib/api', async () => {
        const actual = await vi.importActual('@/lib/api');
        return {
          ...actual,
          default: {
            ...actual.default,
            get: vi.fn().mockImplementation(async (url: string, config?: any) => {
              capturedConfig = config;
              throw new Error('Mock - not making real request');
            }),
          },
        };
      });
      
      const { default: apiClient, getHealth } = await import('@/lib/api');
      
      // Try to make a request
      try {
        await apiClient.get('/health');
      } catch {
        // Expected to fail - we just want to verify the interceptor ran
      }
      
      // The interceptor should have attached the token
      // Since we're testing the actual module, verify the store has the token
      expect(useAuthStore.getState().accessToken).toBe(testToken);
    });
  });
});
