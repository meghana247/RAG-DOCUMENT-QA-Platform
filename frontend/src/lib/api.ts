const BASE = '/api'
const TOKEN_KEY = 'docqa.token'

export const tokenStore = {
  get: () => localStorage.getItem(TOKEN_KEY),
  set: (t: string) => localStorage.setItem(TOKEN_KEY, t),
  clear: () => localStorage.removeItem(TOKEN_KEY),
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message)
  }
}

let onUnauthorized: () => void = () => {}
export const setUnauthorizedHandler = (fn: () => void) => (onUnauthorized = fn)

async function toError(res: Response): Promise<ApiError> {
  const body = await res.json().catch(() => null)
  // Backend errors arrive as {error: {code, message}}; FastAPI validation errors as {detail: [...]}.
  const err = body?.error
  const detail = Array.isArray(body?.detail) ? body.detail[0]?.msg : body?.detail
  return new ApiError(res.status, err?.code ?? 'error', err?.message ?? detail ?? res.statusText)
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = tokenStore.get()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')

  const res = await fetch(BASE + path, { ...init, headers })
  if (res.status === 401 && token) onUnauthorized()
  if (!res.ok) throw await toError(res)
  return (res.status === 204 ? undefined : await res.json()) as T
}

/** POST that answers with server-sent events. EventSource cannot POST or send headers, so parse the stream by hand. */
export async function stream(
  path: string,
  body: unknown,
  onEvent: (event: string, data: any) => void,
  signal?: AbortSignal,
) {
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${tokenStore.get()}` },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) throw await toError(res)

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const event = /^event: (.*)$/m.exec(frame)?.[1] ?? 'message'
      const data = /^data: (.*)$/m.exec(frame)?.[1]
      if (data) onEvent(event, JSON.parse(data))
    }
  }
}

// ---- types mirroring app/schemas ----
export type Page<T> = { items: T[]; total: number; limit: number; offset: number }
export type User = { id: string; email: string; role: string; is_active: boolean; created_at: string }
export type DocStatus = 'pending' | 'processing' | 'ready' | 'failed'
export type Doc = {
  id: string
  filename: string
  content_type: string
  size_bytes: number
  status: DocStatus
  error_message: string | null
  total_chunks: number
  page_count: number | null
  created_at: string
}
export type Citation = {
  chunk_id: string
  document_id: string
  filename: string
  chunk_index: number
  page_number: number | null
  section: string | null
  content: string
  score: number
}
export type Session = { id: string; document_id: string | null; title: string; created_at: string }
export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources: Citation[]
  latency_ms: number | null
  created_at: string
}

export const api = {
  login: (email: string, password: string) =>
    request<{ access_token: string }>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  register: (email: string, password: string) =>
    request<User>('/auth/register', { method: 'POST', body: JSON.stringify({ email, password }) }),
  me: () => request<User>('/auth/me'),

  documents: () => request<Page<Doc>>('/documents?limit=100'),
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<{ document: Doc; duplicate: boolean; message: string }>('/documents/upload', {
      method: 'POST',
      body: form,
    })
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: 'DELETE' }),

  sessions: () => request<Page<Session>>('/chat/sessions?limit=100'),
  createSession: (title: string, document_id: string | null) =>
    request<Session>('/chat/sessions', { method: 'POST', body: JSON.stringify({ title, document_id }) }),
  deleteSession: (id: string) => request<void>(`/chat/sessions/${id}`, { method: 'DELETE' }),
  messages: (id: string) => request<Page<Message>>(`/chat/sessions/${id}/messages?limit=100`),
}
