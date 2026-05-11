/**
 * Centralized API utilities.
 *
 * In production: HttpOnly cookie (hf_access_token) is sent automatically.
 * In development: auth is bypassed on the backend.
 */

import { triggerLogin } from '@/hooks/useAuth';

export interface ApiUploadProgress {
  loaded: number;
  total: number | null;
  percent: number | null;
}

async function handleUnauthorized(response: Response): Promise<void> {
  if (response.status !== 401) return;
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

  await handleUnauthorized(response);

  return response;
}

function headersFromXhr(rawHeaders: string): Headers {
  const headers = new Headers();
  rawHeaders.trim().split(/[\r\n]+/).forEach((line) => {
    const separator = line.indexOf(':');
    if (separator <= 0) return;
    headers.append(
      line.slice(0, separator).trim(),
      line.slice(separator + 1).trim(),
    );
  });
  return headers;
}

export async function apiUpload(
  path: string,
  formData: FormData,
  options: { onProgress?: (progress: ApiUploadProgress) => void } = {},
): Promise<Response> {
  return new Promise<Response>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', path);
    xhr.withCredentials = true;
    xhr.upload.onprogress = (event) => {
      const total = event.lengthComputable ? event.total : null;
      const percent = total
        ? Math.min(100, Math.round((event.loaded / total) * 100))
        : null;
      options.onProgress?.({ loaded: event.loaded, total, percent });
    };
    xhr.onerror = () => reject(new Error('Network error while uploading.'));
    xhr.onabort = () => reject(new Error('Dataset upload was canceled.'));
    xhr.onload = () => {
      const response = new Response(xhr.responseText, {
        status: xhr.status,
        statusText: xhr.statusText,
        headers: headersFromXhr(xhr.getAllResponseHeaders()),
      });
      handleUnauthorized(response).then(() => resolve(response)).catch(reject);
    };
    xhr.send(formData);
  });
}
