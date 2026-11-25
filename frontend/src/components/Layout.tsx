import { NavLink, Outlet } from 'react-router-dom'
import { BarChart3, FileText, LogOut, MessagesSquare, Sparkles } from 'lucide-react'
import { useAuth } from '../lib/auth'
import { cx } from './ui'

const nav = [
  { to: '/chat', label: 'Ask', icon: MessagesSquare },
  { to: '/documents', label: 'Documents', icon: FileText },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
]

export function Layout() {
  const { user, logout } = useAuth()
  return (
    <div className="flex h-screen">
      <aside className="flex w-60 shrink-0 flex-col bg-zinc-950 text-zinc-300">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <div className="grid size-8 place-items-center rounded-lg bg-brand-600 text-white">
            <Sparkles className="size-4" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white">DocQA</p>
            <p className="text-[11px] text-zinc-500">Grounded RAG</p>
          </div>
        </div>
        <nav className="flex-1 space-y-0.5 px-3">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cx(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition',
                  isActive ? 'bg-zinc-800 text-white' : 'hover:bg-zinc-900 hover:text-white',
                )
              }
            >
              <Icon className="size-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-zinc-800 p-3">
          <div className="flex items-center gap-3 rounded-lg px-2 py-2">
            <div className="grid size-8 place-items-center rounded-full bg-zinc-800 text-xs font-semibold text-white uppercase">
              {user?.email[0]}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm text-white">{user?.email}</p>
              <p className="text-[11px] text-zinc-500 capitalize">{user?.role}</p>
            </div>
            <button onClick={logout} className="rounded-md p-1.5 hover:bg-zinc-800" title="Sign out">
              <LogOut className="size-4" />
            </button>
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
