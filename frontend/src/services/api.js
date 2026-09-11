export const API_URL = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_URL || 'https://documind-ai-97t5.onrender.com').replace(/\/+$/, '');

function redactHeaders(headers) {
  const out = {};
  for (const [key, value] of Object.entries(headers || {})) {
    if (/authorization/i.test(key)) {
      out[key] = `Bearer <redacted:${String(value).length} chars>`;
    } else {
      out[key] = value;
    }
  }
  return out;
}

export function friendlyNetworkMessage(err, url) {
  const isNetworkFailure =
    err instanceof TypeError ||
    /failed to fetch|fetch failed|networkerror|load failed/i.test(String((err && err.message) || err));
  if (isNetworkFailure) {
    return (
      `Could not reach the server at ${url || API_URL}. ` +
      `If the backend is waking up from a free-tier cold start, it can take up to a minute ` +
      `— please wait a moment and try again.`
    );
  }
  return (err && err.message) || 'Unexpected network error';
}

async function fetchWithDiagnostics(url, options = {}) {
  const method = (options.method || 'GET').toUpperCase();
  const headers = redactHeaders(options.headers || {});
  const signalInfo = options.signal ? { aborted: options.signal.aborted } : undefined;

  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      console.error(
        `[API] Non-OK response for ${method} ${url}`,
        { status: response.status, statusText: response.statusText, headers, signalInfo }
      );
    }
    return response;
  } catch (err) {
    console.error(
      `[API] Network request failed for ${method} ${url}`,
      { headers, signalInfo, error: err }
    );
    throw err;
  }
}

export function getAuthToken() {
  return localStorage.getItem('kuerycore_token');
}

export function setAuthToken(token) {
  if (token) {
    localStorage.setItem('kuerycore_token', token);
  } else {
    localStorage.removeItem('kuerycore_token');
  }
}

export function removeAuthToken() {
  localStorage.removeItem('kuerycore_token');
  localStorage.removeItem('kuerycore_refresh_token');
}

export function setAuthData(data) {
  if (data.access_token) setAuthToken(data.access_token);
  if (data.refresh_token) {
    localStorage.setItem('kuerycore_refresh_token', data.refresh_token);
  } else {
    localStorage.removeItem('kuerycore_refresh_token');
  }
}

export async function refreshAccessToken() {
  const refreshToken = localStorage.getItem('kuerycore_refresh_token');
  if (!refreshToken) throw new Error('No refresh token');

  let response;
  try {
    response = await fetchWithDiagnostics(`${API_URL}/api/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  } catch (err) {
    throw new Error(friendlyNetworkMessage(err, `${API_URL}/api/auth/refresh`));
  }

  if (!response.ok) {
    removeAuthToken();
    throw new Error('Session expired. Please log in again.');
  }

  const data = await response.json();
  setAuthData(data);
  return data.access_token;
}

export async function authedFetch(url, options = {}, retry = true) {
  const headers = { ...(options.headers || {}) };
  const token = getAuthToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  let response;
  try {
    response = await fetchWithDiagnostics(url, { ...options, headers });
  } catch (err) {
    throw new Error(friendlyNetworkMessage(err, url));
  }

  if (response.status === 401 && retry) {
    try {
      const newToken = await refreshAccessToken();
      headers['Authorization'] = `Bearer ${newToken}`;
      response = await fetchWithDiagnostics(url, { ...options, headers });
    } catch (e) {
      window.dispatchEvent(new CustomEvent('auth-expired'));
      throw e;
    }
  }

  return response;
}

export async function loginUser(email, password, signal) {
  let response;
  try {
    response = await fetchWithDiagnostics(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      signal,
    });
  } catch (err) {
    throw new Error(friendlyNetworkMessage(err, `${API_URL}/api/auth/login`));
  }

  if (!response.ok) {
    let errorMsg = 'Login failed';
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  const data = await response.json();
  setAuthData(data);
  return data;
}

export async function signupUser(email, password, signal) {
  let response;
  try {
    response = await fetchWithDiagnostics(`${API_URL}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      signal,
    });
  } catch (err) {
    throw new Error(friendlyNetworkMessage(err, `${API_URL}/api/auth/signup`));
  }

  if (!response.ok) {
    let errorMsg = 'Signup failed';
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch {}
    // 400 "already exists" and 409 Conflict both surface clearly
    if (response.status === 409 || (response.status === 400 && /already exists/i.test(errorMsg))) {
      throw new Error('An account with this email already exists. Try signing in instead.');
    }
    throw new Error(errorMsg);
  }

  const data = await response.json();
  setAuthData(data);
  return data;
}

export async function requestPasswordReset(email, signal) {
  let response;
  try {
    response = await fetchWithDiagnostics(`${API_URL}/api/auth/forgot-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
      signal,
    });
  } catch (err) {
    throw new Error(friendlyNetworkMessage(err, `${API_URL}/api/auth/forgot-password`));
  }

  if (!response.ok) {
    let errorMsg = 'Failed to send reset email';
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json(); // { message: "If an account exists..." }
}

export async function resetPassword(token, newPassword, signal) {
  let response;
  try {
    response = await fetchWithDiagnostics(`${API_URL}/api/auth/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, new_password: newPassword }),
      signal,
    });
  } catch (err) {
    throw new Error(friendlyNetworkMessage(err, `${API_URL}/api/auth/reset-password`));
  }

  if (!response.ok) {
    let errorMsg = 'Password reset failed';
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json(); // { message: "Password updated successfully..." }
}

export async function fetchCurrentUser(signal = null) {
  const response = await authedFetch(`${API_URL}/api/auth/me`, { signal });
  if (!response.ok) {
    removeAuthToken();
    throw new Error('Unauthorized');
  }
  return response.json();
}

export async function fetchDocuments() {
  const response = await authedFetch(`${API_URL}/api/documents`);
  if (!response.ok) {
    throw new Error('Failed to fetch documents');
  }
  return response.json();
}

export async function getDocument(documentId) {
  const response = await authedFetch(`${API_URL}/api/documents/${documentId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch document status');
  }
  return response.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await authedFetch(`${API_URL}/api/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMsg = 'Upload failed';
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch {}
    throw new Error(errorMsg);
  }

  return response.json();
}

export async function deleteDocument(documentId) {
  const response = await authedFetch(`${API_URL}/api/documents/${documentId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to delete document');
  }
}

export function buildDocumentFileRequest(documentId) {
  const headers = {};
  const token = getAuthToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return {
    url: `${API_URL}/api/documents/${documentId}/file`,
    httpHeaders: headers,
  };
}

export async function fetchConversations() {
  const response = await authedFetch(`${API_URL}/api/conversations`);
  if (!response.ok) {
    throw new Error('Failed to fetch conversation sessions');
  }
  const data = await response.json();
  return data.conversations || [];
}

export async function fetchConversationDetails(conversationId, signal = null) {
  const response = await authedFetch(`${API_URL}/api/conversations/${conversationId}`, {
    signal,
  });
  if (!response.ok) {
    throw new Error('Failed to fetch conversation history');
  }
  return response.json();
}

export async function deleteConversation(conversationId) {
  const response = await authedFetch(`${API_URL}/api/conversations/${conversationId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const errorMessage = errorData.detail || 'Conversation not found or could not be deleted.';
    throw new Error(errorMessage);
  }
}

/**
 * Pre-warm the backend by pinging /api/health silently in the background.
 * Triggers Render spin-up early before user even clicks anything.
 */
export function warmUpBackend() {
  try {
    fetch(`${API_URL}/api/health`, { method: 'GET', keepalive: true }).catch(() => {});
  } catch (e) {
    // Ignore background pre-warming failures
  }
}
