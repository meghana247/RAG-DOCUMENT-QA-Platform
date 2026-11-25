import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ArrowUp, BookOpen, FileText, MessageSquarePlus, Sparkles, Trash2, X } from 'lucide-react'
import { api, stream, type Citation, type Doc, type Message, type Session } from '../lib/api'
import { Badge, cx } from '../components/ui'
import { Answer } from '../components/Answer'

type LiveMessage = Omit<Message, 'created_at'> & { streaming?: boolean }

const suggestions = [
  'Summarise the key points of this document',
  'What are the main risks it describes?',
  'List any definitions with their page numbers',
]

export function Chat() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [active, setActive] = useState<Session | null>(null)
  const [messages, setMessages] = useState<LiveMessage[]>([])
  const [docs, setDocs] = useState<Doc[]>([])
  const [scope, setScope] = useState<string>('')
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [citation, setCitation] = useState<Citation | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.sessions().then((p) => setSessions(p.items))
    api.documents().then((p) => setDocs(p.items.filter((d) => d.status === 'ready')))
  }, [])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const open = async (s: Session) => {
    setActive(s)
    setCitation(null)
    setScope(s.document_id ?? '')
    setMessages((await api.messages(s.id)).items)
  }

  const newChat = () => {
    setActive(null)
    setMessages([])
    setCitation(null)
  }

  const removeSession = async (s: Session) => {
    await api.deleteSession(s.id)
    setSessions((list) => list.filter((x) => x.id !== s.id))
    if (active?.id === s.id) newChat()
  }

  const ask = async (question: string) => {
    if (!question.trim() || busy) return
    setBusy(true)
    setError('')
    setInput('')

    let session = active
    try {
      if (!session) {
        session = await api.createSession(question.slice(0, 60), scope || null)
        setActive(session)
        setSessions((list) => [session!, ...list])
      }

      const answerId = crypto.randomUUID()
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), role: 'user', content: question, sources: [], latency_ms: null },
        { id: answerId, role: 'assistant', content: '', sources: [], latency_ms: null, streaming: true },
      ])
      const patch = (fn: (m: LiveMessage) => LiveMessage) =>
        setMessages((list) => list.map((m) => (m.id === answerId ? fn(m) : m)))

      const started = performance.now()
      await stream(`/chat/sessions/${session.id}/messages/stream`, { content: question }, (event, data) => {
        if (event === 'sources') patch((m) => ({ ...m, sources: data.sources }))
        if (event === 'token') patch((m) => ({ ...m, content: m.content + data.delta }))
        if (event === 'error') setError(data.message)
      })
      patch((m) => ({ ...m, streaming: false, latency_ms: Math.round(performance.now() - started) }))
    } catch (err) {
      setError((err as Error).message)
      setMessages((m) => m.filter((x) => !x.streaming))
    } finally {
      setBusy(false)
    }
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    ask(input)
  }

  const scopeName = docs.find((d) => d.id === (active?.document_id ?? scope))?.filename

  return (
    <div className="flex h-full">
      {/* sessions */}
      <div className="flex w-64 shrink-0 flex-col border-r border-zinc-200 bg-white">
        <div className="p-3">
          <button onClick={newChat} className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium ring-1 ring-zinc-200 hover:bg-zinc-50">
            <MessageSquarePlus className="size-4" /> New conversation
          </button>
        </div>
        <div className="flex-1 space-y-0.5 overflow-y-auto px-3 pb-3">
          {sessions.map((s) => (
            <div
              key={s.id}
              onClick={() => open(s)}
              className={cx('group flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm', active?.id === s.id ? 'bg-brand-50 text-brand-700' : 'text-zinc-700 hover:bg-zinc-50')}
            >
              <span className="flex-1 truncate">{s.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  removeSession(s)
                }}
                className="opacity-0 group-hover:opacity-100 hover:text-red-600"
              >
                <Trash2 className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* conversation */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-14 items-center justify-between border-b border-zinc-200 bg-white/80 px-6 backdrop-blur">
          <p className="truncate text-sm font-medium">{active?.title ?? 'New conversation'}</p>
          {active ? (
            <Badge tone="brand">
              <BookOpen className="size-3" /> {scopeName ?? 'All documents'}
            </Badge>
          ) : (
            <select value={scope} onChange={(e) => setScope(e.target.value)} className="rounded-lg border-0 bg-white py-1.5 pr-8 pl-3 text-sm ring-1 ring-zinc-200 focus:ring-2 focus:ring-brand-500">
              <option value="">Search all documents</option>
              {docs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center px-6 text-center">
              <div className="grid size-12 place-items-center rounded-2xl bg-brand-600 text-white shadow-lg shadow-brand-600/25">
                <Sparkles className="size-6" />
              </div>
              <h2 className="mt-4 text-lg font-semibold">Ask anything about your documents</h2>
              <p className="mt-1 text-sm text-zinc-500">
                {docs.length ? `${docs.length} document${docs.length > 1 ? 's' : ''} indexed. Answers cite the passages they use.` : 'Upload a document first on the Documents page.'}
              </p>
              <div className="mt-6 grid w-full gap-2">
                {suggestions.map((s) => (
                  <button key={s} onClick={() => ask(s)} disabled={!docs.length} className="rounded-lg bg-white px-4 py-2.5 text-left text-sm text-zinc-700 ring-1 ring-zinc-200 hover:ring-brand-500 disabled:opacity-50">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6 px-6 py-8">
              {messages.map((m) =>
                m.role === 'user' ? (
                  <div key={m.id} className="flex justify-end">
                    <p className="max-w-[80%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm text-white">{m.content}</p>
                  </div>
                ) : (
                  <div key={m.id} className="flex gap-3">
                    <div className="grid size-8 shrink-0 place-items-center rounded-lg bg-zinc-900 text-white">
                      <Sparkles className="size-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className={cx('text-sm leading-relaxed text-zinc-800', m.streaming && 'caret')}>
                        {m.content ? <Answer text={m.content} sources={m.sources} onCite={setCitation} /> : m.streaming && <span className="text-zinc-400">Searching your documents…</span>}
                      </div>
                      {m.sources.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {m.sources.map((c, i) => (
                            <button
                              key={c.chunk_id}
                              onClick={() => setCitation(c)}
                              className={cx('flex items-center gap-1.5 rounded-md px-2 py-1 text-xs ring-1', citation?.chunk_id === c.chunk_id ? 'bg-brand-50 text-brand-700 ring-brand-500' : 'bg-white text-zinc-600 ring-zinc-200 hover:ring-brand-500')}
                            >
                              <span className="font-semibold">[{i + 1}]</span>
                              <span className="max-w-40 truncate">{c.filename}</span>
                              {c.page_number && <span className="text-zinc-400">p.{c.page_number}</span>}
                            </button>
                          ))}
                        </div>
                      )}
                      {!m.streaming && m.latency_ms != null && <p className="mt-2 text-[11px] text-zinc-400">{(m.latency_ms / 1000).toFixed(1)}s</p>}
                    </div>
                  </div>
                ),
              )}
              <div ref={bottom} />
            </div>
          )}
        </div>

        <div className="border-t border-zinc-200 bg-white p-4">
          {error && <p className="mx-auto mb-2 max-w-3xl text-sm text-red-600">{error}</p>}
          <form onSubmit={submit} className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl bg-zinc-50 p-2 ring-1 ring-zinc-200 focus-within:ring-2 focus-within:ring-brand-500">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) submit(e)
              }}
              rows={1}
              placeholder="Ask a question…"
              className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm focus:outline-none"
            />
            <button disabled={busy || !input.trim()} className="grid size-8 place-items-center rounded-lg bg-brand-600 text-white disabled:opacity-40">
              <ArrowUp className="size-4" />
            </button>
          </form>
        </div>
      </div>

      {/* citation inspector */}
      {citation && (
        <aside className="w-96 shrink-0 overflow-y-auto border-l border-zinc-200 bg-white">
          <div className="flex h-14 items-center justify-between border-b border-zinc-200 px-5">
            <p className="text-sm font-medium">Source passage</p>
            <button onClick={() => setCitation(null)} className="rounded-md p-1 hover:bg-zinc-100">
              <X className="size-4" />
            </button>
          </div>
          <div className="space-y-4 p-5">
            <div className="flex items-start gap-3">
              <FileText className="mt-0.5 size-4 text-zinc-400" />
              <div>
                <p className="text-sm font-medium">{citation.filename}</p>
                <p className="text-xs text-zinc-500">
                  {citation.page_number ? `Page ${citation.page_number} · ` : ''}Chunk {citation.chunk_index}
                  {citation.section ? ` · ${citation.section}` : ''}
                </p>
              </div>
            </div>
            <div className="flex justify-between rounded-lg bg-zinc-50 px-3 py-2 text-xs text-zinc-500 ring-1 ring-zinc-100">
              {/* With reranking on, this is the cross-encoder's logit rather than a 0–1 similarity. */}
              <span>{citation.score > 1 ? 'Rerank score' : 'Similarity'}</span>
              <span className="font-medium text-zinc-800 tabular-nums">{citation.score.toFixed(3)}</span>
            </div>
            <p className="rounded-lg bg-zinc-50 p-4 text-sm leading-relaxed whitespace-pre-wrap text-zinc-700 ring-1 ring-zinc-100">{citation.content}</p>
          </div>
        </aside>
      )}
    </div>
  )
}
