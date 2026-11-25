import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import { AlertCircle, CheckCircle2, FileText, Loader2, Trash2, UploadCloud } from 'lucide-react'
import { api, type Doc, type DocStatus } from '../lib/api'
import { Badge, Card, EmptyState, PageHeader, cx } from '../components/ui'

const statusBadge: Record<DocStatus, { tone: 'gray' | 'blue' | 'green' | 'red'; label: string }> = {
  pending: { tone: 'gray', label: 'Queued' },
  processing: { tone: 'blue', label: 'Indexing' },
  ready: { tone: 'green', label: 'Ready' },
  failed: { tone: 'red', label: 'Failed' },
}

const formatSize = (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.ceil(b / 1e3)} KB`)
const ACCEPT = '.pdf,.docx,.txt,.md'

export function Documents() {
  const [docs, setDocs] = useState<Doc[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const load = useCallback(() => api.documents().then((p) => setDocs(p.items)).finally(() => setLoading(false)), [])

  useEffect(() => {
    load()
  }, [load])

  // Ingestion runs after the 202, so poll while anything is still being indexed.
  const inFlight = docs.some((d) => d.status === 'pending' || d.status === 'processing')
  useEffect(() => {
    if (!inFlight) return
    const t = setInterval(load, 2000)
    return () => clearInterval(t)
  }, [inFlight, load])

  const upload = async (files: FileList | null) => {
    if (!files?.length) return
    setUploading(true)
    setNotice(null)
    try {
      for (const file of Array.from(files)) {
        const res = await api.upload(file)
        setNotice({ ok: true, text: res.duplicate ? `${file.name} was already indexed.` : `${file.name} uploaded — indexing now.` })
      }
      await load()
    } catch (err) {
      setNotice({ ok: false, text: (err as Error).message })
    } finally {
      setUploading(false)
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    upload(e.dataTransfer.files)
  }

  const remove = async (doc: Doc) => {
    if (!confirm(`Delete ${doc.filename}? Its chunks and embeddings are removed too.`)) return
    await api.deleteDocument(doc.id)
    setDocs((d) => d.filter((x) => x.id !== doc.id))
  }

  const totalChunks = docs.reduce((s, d) => s + d.total_chunks, 0)

  return (
    <div className="mx-auto max-w-5xl p-8">
      <PageHeader title="Documents" description="Upload PDFs, Word files or text. They are chunked, embedded and indexed for retrieval." />

      <div className="mb-6 grid grid-cols-3 gap-4">
        {[
          ['Documents', docs.length],
          ['Ready', docs.filter((d) => d.status === 'ready').length],
          ['Indexed chunks', totalChunks],
        ].map(([label, value]) => (
          <Card key={label} className="p-4">
            <p className="text-xs font-medium text-zinc-500">{label}</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
          </Card>
        ))}
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        className={cx(
          'mb-4 flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition',
          dragging ? 'border-brand-500 bg-brand-50' : 'border-zinc-300 bg-white hover:border-brand-500/60',
        )}
      >
        {uploading ? <Loader2 className="size-8 animate-spin text-brand-600" /> : <UploadCloud className="size-8 text-brand-600" />}
        <p className="mt-3 text-sm font-medium">{uploading ? 'Uploading…' : 'Drop files here or click to browse'}</p>
        <p className="mt-1 text-xs text-zinc-500">PDF, DOCX, TXT or MD · up to 25 MB · duplicates are detected by content hash</p>
        <input ref={input} type="file" accept={ACCEPT} multiple hidden onChange={(e) => upload(e.target.files)} />
      </div>

      {notice && (
        <div className={cx('mb-4 flex items-center gap-2 rounded-lg px-3 py-2 text-sm', notice.ok ? 'bg-emerald-50 text-emerald-800' : 'bg-red-50 text-red-700')}>
          {notice.ok ? <CheckCircle2 className="size-4" /> : <AlertCircle className="size-4" />}
          {notice.text}
        </div>
      )}

      <Card>
        {loading ? (
          <div className="flex justify-center py-14">
            <Loader2 className="size-5 animate-spin text-zinc-400" />
          </div>
        ) : docs.length === 0 ? (
          <EmptyState icon={<FileText className="size-5" />} title="No documents yet" hint="Upload a file to start asking questions about it." />
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-100 text-left text-xs text-zinc-500">
                <th className="px-5 py-3 font-medium">Name</th>
                <th className="px-5 py-3 font-medium">Status</th>
                <th className="px-5 py-3 font-medium">Pages</th>
                <th className="px-5 py-3 font-medium">Chunks</th>
                <th className="px-5 py-3 font-medium">Size</th>
                <th className="px-5 py-3 font-medium">Uploaded</th>
                <th />
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {docs.map((d) => (
                <tr key={d.id} className="group hover:bg-zinc-50/60">
                  <td className="px-5 py-3">
                    <div className="flex items-center gap-3">
                      <FileText className="size-4 shrink-0 text-zinc-400" />
                      <span className="font-medium">{d.filename}</span>
                    </div>
                    {d.error_message && <p className="mt-1 pl-7 text-xs text-red-600">{d.error_message}</p>}
                  </td>
                  <td className="px-5 py-3">
                    <Badge tone={statusBadge[d.status].tone}>
                      {(d.status === 'processing' || d.status === 'pending') && <Loader2 className="size-3 animate-spin" />}
                      {statusBadge[d.status].label}
                    </Badge>
                  </td>
                  <td className="px-5 py-3 tabular-nums text-zinc-600">{d.page_count ?? '—'}</td>
                  <td className="px-5 py-3 tabular-nums text-zinc-600">{d.total_chunks}</td>
                  <td className="px-5 py-3 tabular-nums text-zinc-600">{formatSize(d.size_bytes)}</td>
                  <td className="px-5 py-3 text-zinc-500">{new Date(d.created_at).toLocaleDateString()}</td>
                  <td className="px-3 py-3 text-right">
                    <button onClick={() => remove(d)} className="rounded-md p-1.5 text-zinc-400 opacity-0 group-hover:opacity-100 hover:bg-red-50 hover:text-red-600">
                      <Trash2 className="size-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
