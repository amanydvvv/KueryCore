import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react';
import AuthModal from '../components/AuthModal';
import * as api from '../services/api';

// Mock the 3D WebGL spheres to avoid Canvas/WebGL dependencies in JSDOM
vi.mock('../components/auth/Interactive3DSpheres', () => ({
  default: () => <div data-testid="interactive-3d-spheres" />,
}));

beforeEach(() => {
  // Mock warmUpBackend to be a no-op so it doesn't fire real fetches in tests
  vi.spyOn(api, 'warmUpBackend').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('AuthModal Component Regressions', () => {
  it('renders distinct 429 rate-limit error message when server responds with 429', async () => {
    const rateLimitError = new Error('Too many requests. Please slow down and try again later.');
    vi.spyOn(api, 'loginUser').mockRejectedValue(rateLimitError);

    const { container } = render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    // Fill in credentials
    fireEvent.change(screen.getByPlaceholderText(/e-mail address/i), {
      target: { value: 'test@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/password/i), {
      target: { value: 'ValidPass1234!' },
    });

    // Click Sign In Submit Button
    const submitBtn = container.querySelector('#auth-submit-btn');
    fireEvent.click(submitBtn);

    // Wait for the rate-limit message to be rendered
    await waitFor(() => {
      const el = screen.getByText(/too many requests\. please slow down and try again later\./i);
      expect(el).toBeTruthy();
    });
  });

  it('double-submit guard blocks rapid double-clicking on sign-in button', async () => {
    let resolveLogin;
    const loginPromise = new Promise((resolve) => {
      resolveLogin = resolve;
    });

    const loginSpy = vi.spyOn(api, 'loginUser').mockImplementation(() => loginPromise);

    const { container } = render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    // Fill in credentials
    fireEvent.change(screen.getByPlaceholderText(/e-mail address/i), {
      target: { value: 'rapid@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/password/i), {
      target: { value: 'ValidPass1234!' },
    });

    const submitBtn = container.querySelector('#auth-submit-btn');

    // Click rapidly multiple times in succession
    fireEvent.click(submitBtn);
    fireEvent.click(submitBtn);
    fireEvent.click(submitBtn);

    // Assert that loginUser API was dispatched EXACTLY ONCE
    expect(loginSpy).toHaveBeenCalledTimes(1);

    // Resolve the in-flight request
    resolveLogin({ access_token: 'tok_123', user_id: 'uid_1', email: 'rapid@example.com' });

    await waitFor(() => {
      expect(loginSpy).toHaveBeenCalledTimes(1);
    });
  });

  it('double-submit guard also protects the sign-up mode', async () => {
    let resolveSignup;
    const signupPromise = new Promise((resolve) => {
      resolveSignup = resolve;
    });

    const signupSpy = vi.spyOn(api, 'signupUser').mockImplementation(() => signupPromise);

    const { container } = render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    // Switch to Sign Up mode via the switcher tab buttons
    const signUpTabs = screen.getAllByRole('button', { name: /sign up/i });
    fireEvent.click(signUpTabs[0]);

    // Fill in credentials
    fireEvent.change(screen.getByPlaceholderText(/e-mail address/i), {
      target: { value: 'newuser@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/password/i), {
      target: { value: 'ValidPass1234!' },
    });

    const submitBtn = container.querySelector('#auth-submit-btn');

    // Click rapidly 3 times
    fireEvent.click(submitBtn);
    fireEvent.click(submitBtn);
    fireEvent.click(submitBtn);

    // Assert that signupUser API was dispatched EXACTLY ONCE
    expect(signupSpy).toHaveBeenCalledTimes(1);

    resolveSignup({ access_token: 'tok_new', user_id: 'uid_2', email: 'newuser@example.com' });

    await waitFor(() => {
      expect(signupSpy).toHaveBeenCalledTimes(1);
    });
  });

  it('forgot-password mode submits and renders confirmation banner', async () => {
    const forgotSpy = vi.spyOn(api, 'requestPasswordReset').mockResolvedValue({
      message: 'If an account exists with this email, password reset instructions have been sent.',
    });

    const { container } = render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    // Click "Forgot password?"
    fireEvent.click(screen.getByText(/forgot password\?/i));

    // Enter email
    fireEvent.change(screen.getByPlaceholderText(/e-mail address/i), {
      target: { value: 'forgot@example.com' },
    });

    // Click Send Reset Link
    const submitBtn = container.querySelector('#forgot-submit-btn');
    fireEvent.click(submitBtn);

    expect(forgotSpy).toHaveBeenCalledTimes(1);
    expect(forgotSpy).toHaveBeenCalledWith('forgot@example.com', expect.any(AbortSignal));

    await waitFor(() => {
      const el = screen.getByText(/if an account exists with this email, password reset instructions have been sent\./i);
      expect(el).toBeTruthy();
    });
  });
});

describe('AuthModal Cold-Start UX Regressions', () => {
  it('calls warmUpBackend on mount to pre-warm the server', () => {
    const warmSpy = vi.spyOn(api, 'warmUpBackend').mockImplementation(() => {});

    render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    expect(warmSpy).toHaveBeenCalledTimes(1);
  });

  it('shows cold-start progress banner after 4 seconds of waiting', async () => {
    vi.useFakeTimers();

    // loginUser never resolves — simulates a slow server
    vi.spyOn(api, 'loginUser').mockImplementation(
      () => new Promise(() => {}) // hangs forever
    );

    const { container } = render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText(/e-mail address/i), {
      target: { value: 'test@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/password/i), {
      target: { value: 'ValidPass1234!' },
    });

    fireEvent.click(container.querySelector('#auth-submit-btn'));

    // Advance 5 seconds using async variant — stops at exactly 5s, no infinite loop
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    const banner = screen.getByText(/waking up cloud server/i);
    expect(banner).toBeTruthy();
  }, 15_000);

  it('shows elapsed seconds in the cold-start banner', async () => {
    vi.useFakeTimers();

    vi.spyOn(api, 'loginUser').mockImplementation(
      () => new Promise(() => {}) // hangs forever
    );

    const { container } = render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText(/e-mail address/i), {
      target: { value: 'test@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/password/i), {
      target: { value: 'ValidPass1234!' },
    });

    fireEvent.click(container.querySelector('#auth-submit-btn'));

    // Advance 6 seconds using async variant — stops at exactly 6s, no infinite loop
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });

    // Banner should show "free tier" messaging with elapsed seconds
    const banner = screen.getByText(/free tier takes/i);
    expect(banner).toBeTruthy();
  }, 15_000);

  it('shows updated timeout message (not old 15s message) on abort', async () => {
    // Immediately reject with an abort error
    const abortErr = new Error('The user aborted a request.');
    abortErr.name = 'AbortError';
    vi.spyOn(api, 'loginUser').mockRejectedValue(abortErr);

    const { container } = render(<AuthModal onAuthSuccess={vi.fn()} onAuthError={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText(/e-mail address/i), {
      target: { value: 'test@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText(/password/i), {
      target: { value: 'ValidPass1234!' },
    });

    fireEvent.click(container.querySelector('#auth-submit-btn'));

    await waitFor(() => {
      // Must show the new 75s-era message, NOT the old "Request timed out" message
      const el = screen.getByText(/server took longer than expected/i);
      expect(el).toBeTruthy();
    });

    // Also verify the OLD message text is gone
    const oldMsg = screen.queryByText(/request timed out\. the server may be starting up/i);
    expect(oldMsg).toBeNull();
  });
});
