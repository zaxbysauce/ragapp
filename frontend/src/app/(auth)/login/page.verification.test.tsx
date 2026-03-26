import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Mock authStore
const mockLogin = vi.fn();
const mockGetState = vi.fn();
const useAuthStoreMock = vi.fn(() => ({
  login: mockLogin,
  isLoading: false,
  mustChangePassword: false,
}));
useAuthStoreMock.getState = mockGetState;

vi.mock('@/stores/authStore', () => ({
  useAuthStore: useAuthStoreMock,
}));

// Mock react-router-dom
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
  Button: ({ children, onClick, disabled, type, 'aria-label': ariaLabel, ...props }: any) => {
    const testId = ariaLabel?.toLowerCase().includes('password') ? 'password-toggle' : 'submit-button';
    return (
      <button data-testid={testId} type={type} onClick={onClick} disabled={disabled} aria-label={ariaLabel} {...props}>
        {children}
      </button>
    );
  },
}));

vi.mock('@/components/ui/input', () => ({
  Input: ({ id, type, value, onChange, disabled, ...props }: any) => (
    <input data-testid={id} type={type} value={value} onChange={onChange} disabled={disabled} {...props} />
  ),
}));

vi.mock('@/components/ui/label', () => ({
  Label: ({ children, htmlFor }: any) => <label htmlFor={htmlFor}>{children}</label>,
}));

vi.mock('lucide-react', () => ({
  Eye: () => <span data-testid="icon-eye">eye</span>,
  EyeOff: () => <span data-testid="icon-eye-off">eye-off</span>,
  Loader2: () => <span data-testid="icon-loader">loader</span>,
  AlertCircle: () => <span data-testid="icon-alert">alert</span>,
}));

import LoginPage from './page';

describe('LoginPage Verification Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLogin.mockClear();
    mockGetState.mockClear();
    mockNavigate.mockClear();
  });

  it('renders username and password fields', () => {
    render(
      <MemoryRouter>
        <LoginPage />
      </MemoryRouter>
    );
    expect(screen.getByTestId('username')).toBeInTheDocument();
    expect(screen.getByTestId('password')).toBeInTheDocument();
  });

  it('password visibility toggle works', async () => {
    render(
      <MemoryRouter>
        <LoginPage />
      </MemoryRouter>
    );

    const passwordInput = screen.getByTestId('password') as HTMLInputElement;
    expect(passwordInput.type).toBe('password');

    const toggleButton = screen.getByTestId('password-toggle');
    await userEvent.click(toggleButton);

    expect(passwordInput.type).toBe('text');
    expect(screen.getByTestId('icon-eye-off')).toBeInTheDocument();

    await userEvent.click(toggleButton);
    expect(passwordInput.type).toBe('password');
    expect(screen.getByTestId('icon-eye')).toBeInTheDocument();
  });

  it('submit button is disabled when fields are empty', () => {
    render(
      <MemoryRouter>
        <LoginPage />
      </MemoryRouter>
    );
    const submitButton = screen.getByTestId('submit-button') as HTMLButtonElement;
    expect(submitButton.disabled).toBe(true);
  });

  it('submit button is disabled when only username is filled', async () => {
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
    render(
      <MemoryRouter>
        <LoginPage />
      </MemoryRouter>
    );
    const usernameInput = screen.getByTestId('username');
    const passwordInput = screen.getByTestId('password');
    const submitButton = screen.getByTestId('submit-button') as HTMLButtonElement;

    expect(submitButton.disabled).toBe(true);
    await userEvent.type(usernameInput, 'testuser');
    await userEvent.type(passwordInput, 'password123');
    expect(submitButton.disabled).toBe(false);
  });

  it('calls authStore.login() with correct arguments', async () => {
    mockLogin.mockResolvedValue(undefined);
    mockGetState.mockReturnValue({ mustChangePassword: false });
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
      expect(mockLogin).toHaveBeenCalledWith('testuser', 'password123');
    });
  });

  it('on successful login with mustChangePassword=true, redirects to /settings?action=change-password', async () => {
    mockLogin.mockResolvedValue(undefined);
    mockGetState.mockReturnValue({ mustChangePassword: true });
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

  it('on successful login with mustChangePassword=false, redirects to /chat', async () => {
    mockLogin.mockResolvedValue(undefined);
    mockGetState.mockReturnValue({ mustChangePassword: false });
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

  it('on login error, displays error message in alert', async () => {
    mockLogin.mockRejectedValue(new Error('Invalid credentials'));
    mockGetState.mockReturnValue({ mustChangePassword: false });
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
      expect(alertElement).toHaveTextContent('Invalid credentials');
    });
  });
});
