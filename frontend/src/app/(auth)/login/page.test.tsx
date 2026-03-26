import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Use vi.hoisted() to define mocks that need to be hoisted
const mocks = vi.hoisted(() => {
  const mockLogin = vi.fn();
  const mockGetState = vi.fn();
  
  return {
    mockLogin,
    mockGetState,
    useAuthStoreMock: Object.assign(
      vi.fn(() => ({
        login: mockLogin,
        isLoading: false,
        mustChangePassword: false,
      })),
      {
        getState: mockGetState,
      }
    ),
  };
});

vi.mock('@/stores/authStore', () => ({
  useAuthStore: mocks.useAuthStoreMock,
}));

// Mock react-router-dom to capture navigate calls
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mock UI components
vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, type, 'aria-busy': ariaBusy, className, 'aria-label': ariaLabel, variant, ...props }: any) => {
    // Give different testid based on aria-label or variant
    const testId = ariaLabel?.toLowerCase().includes('password') || variant === 'ghost' 
      ? 'password-toggle' 
      : 'submit-button';
    return (
      <button
        data-testid={testId}
        type={type}
        onClick={onClick}
        disabled={disabled}
        aria-busy={ariaBusy}
        aria-label={ariaLabel}
        className={className}
        {...props}
      >
        {children}
      </button>
    );
  },
}));

vi.mock('@/components/ui/input', () => ({
  Input: ({ id, type, value, onChange, disabled, 'aria-label': ariaLabel, className, ...props }: any) => (
    <input
      data-testid={id}
      type={type}
      value={value}
      onChange={onChange}
      aria-label={ariaLabel}
      className={className}
      {...props}
      disabled={disabled}
    />
  ),
}));

vi.mock('@/components/ui/card', () => ({
  Card: ({ children, className }: any) => <div data-testid="card" className={className}>{children}</div>,
  CardContent: ({ children, className }: any) => <div className={className}>{children}</div>,
  CardDescription: ({ children }: any) => <p>{children}</p>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <h2>{children}</h2>,
}));

vi.mock('@/components/ui/label', () => ({
  Label: ({ children, htmlFor }: any) => <label htmlFor={htmlFor}>{children}</label>,
}));

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Eye: () => <span data-testid="icon-eye">👁</span>,
  EyeOff: () => <span data-testid="icon-eye-off">👁‍🗨</span>,
  Loader2: () => <span data-testid="icon-loader">⟳</span>,
  AlertCircle: () => <span data-testid="icon-alert">⚠</span>,
}));

import LoginPage from './page';

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.mockLogin.mockClear();
    mocks.mockGetState.mockClear();
    mockNavigate.mockClear();
  });

  describe('Form rendering', () => {
    it('renders username and password fields', () => {
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

      expect(screen.getByTestId('username')).toBeInTheDocument();
      expect(screen.getByTestId('password')).toBeInTheDocument();
    });

    it('renders with RAGApp title', () => {
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

      expect(screen.getByText('RAGApp')).toBeInTheDocument();
    });

    it('renders Sign In button', () => {
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

      expect(screen.getByTestId('submit-button')).toBeInTheDocument();
      expect(screen.getByText('Sign In')).toBeInTheDocument();
    });
  });

  describe('Password visibility toggle', () => {
    it('clicking eye icon toggles password visibility from hidden to shown', async () => {
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

      // Find the toggle button
      const toggleButton = screen.getByTestId('password-toggle');
      expect(toggleButton).not.toBeNull();

      // Click to show password
      await userEvent.click(toggleButton!);
      
      // Password input type should change to text
      await waitFor(() => {
        expect(passwordInput.type).toBe('text');
      });

      // EyeOff icon should now be visible
      expect(screen.getByTestId('icon-eye-off')).toBeInTheDocument();
    });

    it('clicking eye-off icon toggles password visibility from shown to hidden', async () => {
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

      // First, show the password by clicking toggle button
      const showButton = screen.getByTestId('password-toggle');
      await userEvent.click(showButton);

      await waitFor(() => {
        expect(passwordInput.type).toBe('text');
      });

      // Now click again to hide password
      const hideButton = screen.getByTestId('password-toggle');
      await userEvent.click(hideButton);

      await waitFor(() => {
        expect(passwordInput.type).toBe('password');
      });
    });
  });

  describe('Submit button disabled state', () => {
    it('submit button is disabled when fields are empty', () => {
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

    it('submit button is disabled when only username is filled', async () => {
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
      const submitButton = screen.getByTestId('submit-button') as HTMLButtonElement;

      await userEvent.type(usernameInput, 'testuser');
      
      expect(submitButton.disabled).toBe(true);
    });

    it('submit button is enabled when both fields have values', async () => {
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

      // Fill both fields
      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');

      // Button should now be enabled
      expect(submitButton.disabled).toBe(false);
    });

    it('submit button is disabled during loading', async () => {
      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });
      mocks.mockLogin.mockImplementation(() => new Promise(() => {})); // Never resolves
      mocks.useAuthStoreMock.mockReturnValue({
        login: mocks.mockLogin,
        isLoading: true,
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

      // Fill fields first
      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');

      // Submit form
      await userEvent.click(submitButton);

      // Button should be disabled during loading
      await waitFor(() => {
        expect(submitButton.disabled).toBe(true);
      });
    });
  });

  describe('Login submission', () => {
    it('calls authStore.login() with correct username and password', async () => {
      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });
      mocks.mockLogin.mockResolvedValue(undefined);
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
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mocks.mockLogin).toHaveBeenCalledWith('testuser', 'password123');
      });
    });

    it('calls authStore.getState() to check mustChangePassword after login', async () => {
      mocks.mockGetState.mockReturnValue({ mustChangePassword: false });
      mocks.mockLogin.mockResolvedValue(undefined);
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
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mocks.mockGetState).toHaveBeenCalled();
      });
    });
  });

  describe('Redirect behavior', () => {
    it('redirects to /settings?action=change-password when mustChangePassword=true', async () => {
      mocks.mockLogin.mockResolvedValue(undefined);
      mocks.mockGetState.mockReturnValue({ mustChangePassword: true });
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

      const usernameInput = screen.getByTestId('username');
      const passwordInput = screen.getByTestId('password');
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/settings?action=change-password');
      });
    });

    it('redirects to /chat when mustChangePassword=false', async () => {
      mocks.mockLogin.mockResolvedValue(undefined);
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
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/chat');
      });
    });
  });

  describe('Error handling', () => {
    it('displays error message in alert on login failure', async () => {
      mocks.mockLogin.mockRejectedValue(new Error('Invalid credentials'));
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
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'wrongpassword');
      await userEvent.click(submitButton);

      await waitFor(() => {
        const errorElement = screen.getByText('Invalid credentials');
        expect(errorElement).toBeInTheDocument();
      });
    });

    it('error alert has proper accessibility attributes', async () => {
      mocks.mockLogin.mockRejectedValue(new Error('Test error'));
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
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'wrongpassword');
      await userEvent.click(submitButton);

      await waitFor(() => {
        const alertElement = screen.getByRole('alert');
        expect(alertElement).toBeInTheDocument();
        expect(alertElement).toHaveAttribute('aria-live', 'polite');
      });
    });

    it('error alert includes AlertCircle icon', async () => {
      mocks.mockLogin.mockRejectedValue(new Error('Network error'));
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
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'wrongpassword');
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByTestId('icon-alert')).toBeInTheDocument();
      });
    });

    it('shows generic "Login failed" for non-Error exceptions', async () => {
      mocks.mockLogin.mockRejectedValue('string error');
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
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'wrongpassword');
      await userEvent.click(submitButton);

      await waitFor(() => {
        const errorElement = screen.getByText('Login failed');
        expect(errorElement).toBeInTheDocument();
      });
    });

    it('error is cleared on subsequent form submission', async () => {
      // First login fails
      mocks.mockLogin
        .mockRejectedValueOnce(new Error('First error'))
        .mockResolvedValue(undefined);
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
      const submitButton = screen.getByTestId('submit-button');

      // First submission fails
      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'wrongpassword');
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('First error')).toBeInTheDocument();
      });

      // Second submission succeeds
      await userEvent.clear(passwordInput);
      await userEvent.type(passwordInput, 'correctpassword');
      await userEvent.click(submitButton);

      // Error should be cleared
      expect(screen.queryByText('First error')).not.toBeInTheDocument();
    });
  });

  describe('Loading state UI', () => {
    it('shows spinner when form is submitted and loading', async () => {
      mocks.mockLogin.mockImplementation(() => new Promise(() => {})); // Never resolves
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

      // Fill the form
      const usernameInput = screen.getByTestId('username');
      const passwordInput = screen.getByTestId('password');
      const submitButton = screen.getByTestId('submit-button');

      await userEvent.type(usernameInput, 'testuser');
      await userEvent.type(passwordInput, 'password123');

      // Submit form to trigger loading state
      await userEvent.click(submitButton);

      // Should show loading text
      await waitFor(() => {
        expect(screen.getByText('Signing in...')).toBeInTheDocument();
      });
      expect(screen.getByTestId('icon-loader')).toBeInTheDocument();
    });
  });
});
