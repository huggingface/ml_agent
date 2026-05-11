/**
 * Centralized API utilities.
 *
 * In production: HttpOnly cookie (hf_access_token) is sent automatically.
 * In development: auth is bypassed on the backend.
 */

import { triggerLogin } from '@/hooks/useAuth';

/** Wrapper around fetch with credentials and common headers. */
export async function apiFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(options.headers);
  const isFormData = options.body instanceof FormData;
  if (!isFormData && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(path, {
    ...options,
    headers,
    credentials: 'include', // Send cookies with every request
  });

  // Handle 401 — redirect to login
  if (response.status === 401) {
    try {
      const authStatus = await fetch('/auth/status', { credentials: 'include' });
      const data = await authStatus.json();
      if (data.auth_enabled) {
        triggerLogin();
        throw new Error('Authentication required — redirecting to login.');
      }
    } catch (e) {
      if (e instanceof Error && e.message.includes('redirecting')) throw e;
    }
  }

  return response;
}
