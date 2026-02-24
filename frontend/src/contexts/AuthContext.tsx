import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { getHealth } from "@/lib/api";

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (apiKey: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const STORAGE_KEY = "kv_api_key";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setIsAuthenticated(false);
  };

  const checkAuth = async (): Promise<boolean> => {
    const apiKey = localStorage.getItem(STORAGE_KEY);
    if (!apiKey) {
      setIsAuthenticated(false);
      setIsLoading(false);
      return false;
    }

    try {
      // Verify the API key by making a health check
      await getHealth();
      setIsAuthenticated(true);
      setIsLoading(false);
      return true;
    } catch {
      // API key is invalid
      localStorage.removeItem(STORAGE_KEY);
      setIsAuthenticated(false);
      setIsLoading(false);
      return false;
    }
  };

  const login = async (apiKey: string): Promise<void> => {
    // Store the API key first
    localStorage.setItem(STORAGE_KEY, apiKey);
    
    // Verify it's valid
    const isValid = await checkAuth();
    if (!isValid) {
      localStorage.removeItem(STORAGE_KEY);
      throw new Error("Invalid API key");
    }
  };

  useEffect(() => {
    checkAuth();

    // Listen for 401 responses from API
    const handleUnauthorized = () => {
      logout();
    };

    window.addEventListener("auth:unauthorized", handleUnauthorized);
    return () => {
      window.removeEventListener("auth:unauthorized", handleUnauthorized);
    };
  }, []);

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        isLoading,
        login,
        logout,
        checkAuth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
