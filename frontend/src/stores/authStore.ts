import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: number;
  username: string;
  full_name: string;
  role: 'superadmin' | 'admin' | 'member' | 'viewer';
  is_active: boolean;
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  setUser: (user: User | null) => void;
  setAccessToken: (token: string | null) => void;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<boolean>;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      isAuthenticated: false,
      isLoading: true, // Start with loading true to prevent flash of content

      setUser: (user) => set({ user, isAuthenticated: !!user }),
      setAccessToken: (accessToken) => set({ accessToken }),

      login: async (username: string, password: string) => {
        set({ isLoading: true });
        try {
          const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
          });

          if (!response.ok) {
            const error = await response.json().catch(() => ({ message: 'Login failed' }));
            throw new Error(error.message || 'Login failed');
          }

          const data = await response.json();
          set({ accessToken: data.access_token });

          // Fetch user profile
          const userResponse = await fetch('/api/auth/me', {
            headers: { Authorization: `Bearer ${data.access_token}` },
          });

          if (!userResponse.ok) {
            throw new Error('Failed to fetch user profile');
          }

          const user = await userResponse.json();
          set({ user, isAuthenticated: true });
        } finally {
          set({ isLoading: false });
        }
      },

      logout: async () => {
        const { accessToken } = get();
        try {
          if (accessToken) {
            await fetch('/api/auth/logout', {
              method: 'POST',
              headers: { Authorization: `Bearer ${accessToken}` },
            });
          }
        } catch (error) {
          // Silently fail logout request - still clear local state
          console.error('Logout request failed:', error);
        } finally {
          set({ user: null, accessToken: null, isAuthenticated: false });
        }
      },

      refreshToken: async () => {
        try {
          const response = await fetch('/api/auth/refresh', {
            method: 'POST',
            credentials: 'include',
          });

          if (!response.ok) {
            set({ user: null, accessToken: null, isAuthenticated: false });
            return false;
          }

          const data = await response.json();
          set({ accessToken: data.access_token });
          return true;
        } catch (error) {
          console.error('Token refresh failed:', error);
          set({ user: null, accessToken: null, isAuthenticated: false });
          return false;
        }
      },

      checkAuth: async () => {
        const { accessToken } = get();

        // Set loading state
        set({ isLoading: true });

        // If no access token, try to refresh
        if (!accessToken) {
          const refreshed = await get().refreshToken();
          if (!refreshed) {
            set({ isAuthenticated: false, isLoading: false });
            return;
          }
        }

        // Validate current token by fetching user profile
        try {
          const token = get().accessToken;
          const response = await fetch('/api/auth/me', {
            headers: { Authorization: `Bearer ${token}` },
          });

          if (response.ok) {
            const user = await response.json();
            set({ user, isAuthenticated: true, isLoading: false });
          } else if (response.status === 401) {
            // Token expired, try refresh
            const refreshed = await get().refreshToken();
            if (refreshed) {
              // Retry fetching user with new token
              const retryResponse = await fetch('/api/auth/me', {
                headers: { Authorization: `Bearer ${get().accessToken}` },
              });
              if (retryResponse.ok) {
                const user = await retryResponse.json();
                set({ user, isAuthenticated: true, isLoading: false });
              } else {
                set({ user: null, accessToken: null, isAuthenticated: false, isLoading: false });
              }
            } else {
              // Refresh failed, not authenticated
              set({ user: null, accessToken: null, isAuthenticated: false, isLoading: false });
            }
          } else {
            set({ user: null, accessToken: null, isAuthenticated: false, isLoading: false });
          }
        } catch (error) {
          console.error('Auth check failed:', error);
          set({ user: null, accessToken: null, isAuthenticated: false, isLoading: false });
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ user: state.user }),
      onRehydrateStorage: () => (state) => {
        // After hydration, if there's no user, we're not authenticated
        // Set isLoading to false so ProtectedRoute can redirect
        if (!state?.user) {
          useAuthStore.setState({ isLoading: false, isAuthenticated: false });
        }
        // If there is a user, checkAuth will validate the session
      },
    }
  )
);
