import { CheckCircle2, Database, FlaskConical, Timer, XCircle } from 'lucide-react'
import report from '@eval/results/retrieval.json'
import thresholds from '@eval/thresholds.json'
import { Badge, Card, PageHeader, cx } from '../components/ui'

type Metrics = Record<string, number | string>
const results = report.results as unknown as { configuration: string; metrics: Metrics; by_category: Record<string, { questions: number; hit_rate?: number; refusal_rate?: number }> }[]

const QUALITY = [
  { key: 'hit_rate', label: 'Hit rate', hint: 'Share of questions where any relevant passage was retrieved' },
  { key: 'recall_at_k', label: 'Recall@k', hint: 'Share of all relevant passages that were retrieved' },
  { key: 'mrr', label: 'MRR', hint: 'How high the first relevant passage ranks' },
  { key: 'ndcg', label: 'nDCG', hint: 'Ranking quality weighted by position' },
  { key: 'refusal_rate', label: 'Refusal rate', hint: 'Unanswerable questions correctly refused' },
] as const

const gate = thresholds.retrieval
const best = results.find((r) => r.configuration === gate.configuration) ?? results[results.length - 1]
const maxOf = (key: string) => Math.max(...results.map((r) => Number(r.metrics[key])))
const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export function Evaluation() {
  const gates = Object.entries(gate.min).map(([k, min]) => ({ key: k, min, value: Number(best.metrics[k]), pass: Number(best.metrics[k]) >= min }))

  return (
    <div className="mx-auto max-w-6xl p-8">
      <PageHeader
        title="Retrieval evaluation"
        description="Each retrieval configuration is scored against a hand-labelled golden set. CI fails the build if the production configuration drops below threshold."
        actions={<Badge tone="gray">rev {report.git_revision} · {new Date(report.generated_at).toLocaleDateString()}</Badge>}
      />

      <div className="mb-6 grid grid-cols-4 gap-4">
        {[
          { icon: FlaskConical, label: 'Golden questions', value: report.questions },
          { icon: Database, label: 'Indexed chunks', value: report.indexed_chunks },
          { icon: CheckCircle2, label: 'Production config', value: best.configuration },
          { icon: Timer, label: 'p95 latency', value: `${Math.round(Number(best.metrics.p95_latency_ms))} ms` },
        ].map(({ icon: Icon, label, value }) => (
          <Card key={label} className="p-4">
            <div className="flex items-center gap-2 text-xs font-medium text-zinc-500">
              <Icon className="size-3.5" /> {label}
            </div>
            <p className="mt-1.5 truncate text-lg font-semibold">{value}</p>
          </Card>
        ))}
      </div>

      <Card className="mb-6 p-5">
        <p className="mb-4 text-sm font-semibold">CI quality gate · {best.configuration}</p>
        <div className="grid grid-cols-4 gap-4">
          {gates.map((g) => (
            <div key={g.key} className={cx('rounded-lg p-3 ring-1', g.pass ? 'bg-emerald-50/50 ring-emerald-200' : 'bg-red-50/50 ring-red-200')}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-zinc-600 uppercase">{g.key.replace('_', ' ')}</span>
                {g.pass ? <CheckCircle2 className="size-4 text-emerald-600" /> : <XCircle className="size-4 text-red-600" />}
              </div>
              <p className="mt-1 text-xl font-semibold tabular-nums">{pct(g.value)}</p>
              <p className="text-xs text-zinc-500">threshold {pct(g.min)}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card className="mb-6 overflow-hidden">
        <div className="border-b border-zinc-100 px-5 py-4">
          <p className="text-sm font-semibold">Configuration comparison</p>
          <p className="text-xs text-zinc-500">Bars are relative to the best score in each column.</p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-zinc-500">
              <th className="px-5 py-3 font-medium">Configuration</th>
              {QUALITY.map((q) => (
                <th key={q.key} className="px-4 py-3 font-medium" title={q.hint}>
                  {q.label}
                </th>
              ))}
              <th className="px-4 py-3 font-medium">p50 latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {results.map((r) => (
              <tr key={r.configuration} className={cx(r === best && 'bg-brand-50/40')}>
                <td className="px-5 py-3 font-medium whitespace-nowrap">
                  {r.configuration} {r === best && <Badge tone="brand">prod</Badge>}
                </td>
                {QUALITY.map((q) => {
                  const v = Number(r.metrics[q.key])
                  const top = v === maxOf(q.key)
                  return (
                    <td key={q.key} className="px-4 py-3">
                      <div className={cx('tabular-nums', top ? 'font-semibold text-zinc-900' : 'text-zinc-600')}>{pct(v)}</div>
                      <div className="mt-1 h-1 w-20 rounded-full bg-zinc-100">
                        <div className={cx('h-full rounded-full', top ? 'bg-brand-600' : 'bg-zinc-300')} style={{ width: `${(v / (maxOf(q.key) || 1)) * 100}%` }} />
                      </div>
                    </td>
                  )
                })}
                <td className="px-4 py-3 tabular-nums text-zinc-600">{Number(r.metrics.p50_latency_ms).toFixed(1)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card className="p-5">
        <p className="mb-1 text-sm font-semibold">Hit rate by question type · {best.configuration}</p>
        <p className="mb-5 text-xs text-zinc-500">Where retrieval is strong and where it still struggles.</p>
        <div className="space-y-3">
          {Object.entries(best.by_category).map(([cat, m]) => {
            // Unanswerable questions have no relevant passage, so they are scored on correct refusals instead.
            const v = m.hit_rate ?? m.refusal_rate ?? 0
            return (
              <div key={cat} className="flex items-center gap-4">
                <span className="w-36 text-sm text-zinc-600 capitalize">
                  {cat.replace('-', ' ')}
                  {m.hit_rate == null && <span className="block text-[11px] text-zinc-400 normal-case">refusal rate</span>}
                </span>
                <div className="h-2.5 flex-1 rounded-full bg-zinc-100">
                  <div className="h-full rounded-full bg-gradient-to-r from-brand-500 to-brand-700" style={{ width: `${v * 100}%` }} />
                </div>
                <span className="w-24 text-right text-sm tabular-nums">
                  {pct(v)} <span className="text-zinc-400">· {m.questions}q</span>
                </span>
              </div>
            )
          })}
        </div>
        <p className="mt-5 text-xs text-zinc-500">Corpus: {report.corpus.join(', ')}</p>
      </Card>
    </div>
  )
}
