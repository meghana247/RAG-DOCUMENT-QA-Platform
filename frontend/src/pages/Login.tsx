import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router-dom'
import { FileSearch, Quote, ShieldCheck, Sparkles } from 'lucide-react'
import { useAuth } from '../lib/auth'
import { Button, Input } from '../components/ui'

const features = [
  { icon: FileSearch, title: 'Hybrid retrieval', text: 'Dense + keyword search fused with RRF, MMR and cross-encoder reranking.' },
  { icon: Quote, title: 'Cited answers', text: 'Every answer links to the exact passages and pages it came from.' },
  { icon: ShieldCheck, title: 'Measured quality', text: 'Retrieval and answer quality scored against a golden set in CI.' },
]

export function Login() {
  const { user, login, register } = useAuth()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to="/chat" replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await (mode === 'login' ? login : register)(email, password)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="relative hidden flex-col justify-between overflow-hidden bg-zinc-950 p-12 text-white lg:flex">
        <div className="absolute -top-40 -left-40 size-[32rem] rounded-full bg-brand-600/30 blur-3xl" />
        <div className="relative flex items-center gap-2.5">
          <div className="grid size-9 place-items-center rounded-lg bg-brand-600">
            <Sparkles className="size-5" />
          </div>
          <span className="text-lg font-semibold">DocQA</span>
        </div>
        <div className="relative">
          <h2 className="max-w-md text-3xl font-semibold tracking-tight">
            Ask your documents. Get answers you can verify.
          </h2>
          <div className="mt-10 space-y-6">
            {features.map(({ icon: Icon, title, text }) => (
              <div key={title} className="flex gap-4">
                <div className="grid size-10 shrink-0 place-items-center rounded-lg bg-white/5 ring-1 ring-white/10">
                  <Icon className="size-5 text-brand-100" />
                </div>
                <div>
                  <p className="font-medium">{title}</p>
                  <p className="text-sm text-zinc-400">{text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        <p className="relative text-xs text-zinc-500">FastAPI · PostgreSQL + pgvector · LangChain · Groq</p>
      </div>

      <div className="flex items-center justify-center p-8">
        <form onSubmit={submit} className="w-full max-w-sm space-y-5">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              {mode === 'login' ? 'Welcome back' : 'Create your account'}
            </h1>
            <p className="mt-1 text-sm text-zinc-500">
              {mode === 'login' ? 'Sign in to continue to your workspace.' : 'Your documents stay private to you.'}
            </p>
          </div>
          <Input label="Email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@company.com" />
          <Input label="Password" type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 8 characters" />
          {error && <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>}
          <Button type="submit" loading={busy} className="w-full">
            {mode === 'login' ? 'Sign in' : 'Create account'}
          </Button>
          <p className="text-center text-sm text-zinc-500">
            {mode === 'login' ? 'New here?' : 'Already have an account?'}{' '}
            <button type="button" className="font-medium text-brand-600 hover:underline" onClick={() => setMode(mode === 'login' ? 'register' : 'login')}>
              {mode === 'login' ? 'Create an account' : 'Sign in'}
            </button>
          </p>
        </form>
      </div>
    </div>
  )
}
