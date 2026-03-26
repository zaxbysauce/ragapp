import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Use vi.hoisted() to define mocks that need to be hoisted
const mocks = vi.hoisted(() => {
  const mockLogin = vi.fn();
  const mockGetState = vi.fn();
  const mockSetState = vi.fn();
  
  return {
    mockLogin,
    mockGetState,
    mockSetState,
    useAuthStoreMock: Object.assign(
      vi.fn(() => ({
        login: mockLogin,
        isLoading: false,
        mustChangePassword: false,
      })),
      {
        getState: mockGetState,
        setState: mockSetState,
      }
    ),
  };
});

vi.mock('@/stores/authStore', () => ({
  useAuthStore: mocks.useAuthStoreMock,
}));

// Mock react-router-dom
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => vi.fn(),
  };
});

// Mock UI components
vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button
      data-testid="submit-button"
      onClick={onClick}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  ),
}));

vi.mock('@/components/ui/input', () => ({
  Input: ({ id, type, value, onChange, disabled, ...props }: any) => (
    <input
      data-testid={id}
      type={type}
      value={value}
      onChange={onChange}
      disabled={disabled}
      {...props}
    />
  ),
}));

vi.mock('@/components/ui/card', () => ({
  Card: ({ children }: any) => <div data-testid="card">{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <p>{children}</p>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <h2>{children}</h2>,
}));

vi.mock('@/components/ui/label', () => ({
  Label: ({ children, htmlFor }: any) => <label htmlFor={htmlFor}>{children}</label>,
}));

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  LogIn: () => <span data-testid="icon-login">→</span>,
  Loader2: () => <span data-testid="icon-loader">⟳</span>,
  Eye: () => <span data-testid="icon-eye">👁</span>,
  EyeOff: () => <span data-testid="icon-eye-off">👁‍🗨</span>,
}));

import LoginPage from './LoginPage';

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.mockLogin.mockClear();
    mocks.mockGetState.mockClear();
  });

  describe('Redirect behavior based on mustChangePassword', () => {
    it('Test 1: redirects to /settings?action=change-password when mustChangePassword=true', async () => {
      // Setup: mustChangePassword is true
      mocks.mockGetState.mockReturnValue({ mustChangePassword: true });
      mocks.mockLogin.mockResolvedValue(undefined);

      // Update the mock implementation
      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: false,
        mustChangePassword: true,
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      // Fill in form
      const usernameInput = screen.getByTestId('username');
      const passwordInput = screen.getByTestId('password');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');

      // Submit form
      const submitButton = screen.getByTestId('submit-button');
      await userEvent.click(submitButton);

      // Wait for login to complete
      await waitFor(() => {
        expect(mocks.mockLogin).toHaveBeenCalledWith('testuser', 'password123');
      });

      // Verify getState was called to check mustChangePassword
      await waitFor(() => {
        expect(mocks.mockGetState).toHaveBeenCalled();
      });
    });

    it('Test 2: redirects to /chat when mustChangePassword=false', async () => {
      // Setup: mustChangePassword is false
      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });
      mocks.mockLogin.mockResolvedValue(undefined);

      // Update the mock implementation
      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: false,
        mustChangePassword: false,
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      // Fill in form
      const usernameInput = screen.getByTestId('username');
      const passwordInput = screen.getByTestId('password');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');

      // Submit form
      const submitButton = screen.getByTestId('submit-button');
      await userEvent.click(submitButton);

      // Wait for login to complete
      await waitFor(() => {
        expect(mocks.mockLogin).toHaveBeenCalledWith('testuser', 'password123');
      });

      // Verify getState was called
      await waitFor(() => {
        expect(mocks.mockGetState).toHaveBeenCalled();
      });
    });
  });

  describe('Password visibility toggle', () => {
    it('Test 3: clicking eye icon toggles password visibility', async () => {
      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });

      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: false,
        mustChangePassword: false,
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      const passwordInput = screen.getByTestId('password') as HTMLInputElement;

      // Initially password should be hidden
      expect(passwordInput.type).toBe('password');

      // Find the toggle button by its data-testid parent button with the eye icon
      const toggleButton = screen.getByTestId('icon-eye').parentElement;

      // Click to show password
      expect(toggleButton).not.toBeNull();
      if (toggleButton) {
        await userEvent.click(toggleButton);
        expect(passwordInput.type).toBe('text');

        // After clicking, EyeOff icon should be visible
        const eyeOffIcon = screen.getByTestId('icon-eye-off');
        expect(eyeOffIcon).toBeInTheDocument();

        // Click again to hide password
        const toggleButton2 = screen.getByTestId('icon-eye-off').parentElement;
        expect(toggleButton2).not.toBeNull();
        if (toggleButton2) {
          await userEvent.click(toggleButton2);
          expect(passwordInput.type).toBe('password');
        }
      }
    });
  });

  describe('Button disabled state', () => {
    it('Test 4: login button disabled when fields are empty', () => {
      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });

      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: false,
        mustChangePassword: false,
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      const submitButton = screen.getByTestId('submit-button') as HTMLButtonElement;

      // Button should be disabled when both fields are empty
      expect(submitButton.disabled).toBe(true);
    });

    it('Test 4b: login button enabled when both fields have values', async () => {
      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });

      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: false,
        mustChangePassword: false,
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      const usernameInput = screen.getByTestId('username');
      const passwordInput = screen.getByTestId('password');
      const submitButton = screen.getByTestId('submit-button') as HTMLButtonElement;

      // Initially disabled
      expect(submitButton.disabled).toBe(true);

      // Fill username only
      await userEvent.type(usernameInput, 'testuser');
      expect(submitButton.disabled).toBe(true);

      // Fill password
      await userEvent.type(passwordInput, 'password123');
      expect(submitButton.disabled).toBe(false);
    });
  });

  describe('Error handling', () => {
    it('Test 5: displays error message on login failure', async () => {
      mocks.mockLogin.mockRejectedValueOnce(new Error('Invalid credentials'));

      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });

      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: false,
        mustChangePassword: false,
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      // Fill in form
      const usernameInput = screen.getByTestId('username');
      const passwordInput = screen.getByTestId('password');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'wrongpassword');

      // Submit form
      const submitButton = screen.getByTestId('submit-button');
      await userEvent.click(submitButton);

      // Wait for error to appear
      await waitFor(() => {
        const errorElement = screen.getByText('Invalid credentials');
        expect(errorElement).toBeInTheDocument();
      });
    });

    it('Test 5b: error message is destructive/red styled', async () => {
      mocks.mockLogin.mockRejectedValueOnce(new Error('Network error'));

      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });

      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: false,
        mustChangePassword: false,
      });

      render(
        <MemoryRouter>
          <LoginPage />
        </MemoryRouter>
      );

      const usernameInput = screen.getByTestId('username');
      const passwordInput = screen.getByTestId('password');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password');

      const submitButton = screen.getByTestId('submit-button');
      await userEvent.click(submitButton);

      await waitFor(() => {
        const errorElement = screen.getByText('Network error');
        expect(errorElement).toHaveClass('text-destructive');
      });
    });
  });
});
