import type { ReactNode } from 'react'
import type { Citation } from '../lib/api'

/** Minimal markdown for model answers: paragraphs, lists, headings, bold, code, and [n] citations. */
export function Answer({ text, sources, onCite }: { text: string; sources: Citation[]; onCite: (c: Citation) => void }) {
  // Some models emit citations as 【1】 or 【1†L1-L4】 instead of the [1] the prompt asks for.
  const clean = text.replace(/【(\d+)(?:†[^】]*)?】/g, '[$1]')

  const inline = (line: string): ReactNode[] =>
    line.split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g).map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) return <strong key={i} className="font-semibold text-zinc-900">{part.slice(2, -2)}</strong>
      if (part.startsWith('`') && part.endsWith('`')) return <code key={i} className="rounded bg-zinc-100 px-1 py-0.5 text-[0.85em]">{part.slice(1, -1)}</code>
      const cite = /^\[(\d+)\]$/.exec(part)
      if (cite) {
        const source = sources[Number(cite[1]) - 1]
        return (
          <button key={i} disabled={!source} onClick={() => source && onCite(source)} title={source?.filename}
            className="mx-0.5 inline-grid h-4 min-w-4 place-items-center rounded bg-brand-100 px-1 align-text-top text-[10px] font-semibold text-brand-700 hover:bg-brand-600 hover:text-white">
            {cite[1]}
          </button>
        )
      }
      return part
    })

  const blocks: ReactNode[] = []
  let list: { ordered: boolean; items: string[] } | null = null
  const flush = () => {
    if (!list) return
    const Tag = list.ordered ? 'ol' : 'ul'
    blocks.push(
      <Tag key={blocks.length} className={list.ordered ? 'list-decimal space-y-1 pl-5' : 'list-disc space-y-1 pl-5'}>
        {list.items.map((item, i) => <li key={i}>{inline(item)}</li>)}
      </Tag>,
    )
    list = null
  }

  for (const raw of clean.split('\n')) {
    const line = raw.trim()
    const bullet = /^[-*•]\s+(.*)/.exec(line)
    const numbered = /^\d+[.)]\s+(.*)/.exec(line)
    if (bullet || numbered) {
      const ordered = !!numbered
      if (list && list.ordered !== ordered) flush()
      list ??= { ordered, items: [] }
      list.items.push((bullet ?? numbered)![1])
      continue
    }
    flush()
    if (!line) continue
    const heading = /^#{1,4}\s+(.*)/.exec(line)
    blocks.push(heading
      ? <p key={blocks.length} className="pt-1 font-semibold text-zinc-900">{inline(heading[1])}</p>
      : <p key={blocks.length}>{inline(line)}</p>)
  }
  flush()

  return <div className="space-y-2.5">{blocks}</div>
}
