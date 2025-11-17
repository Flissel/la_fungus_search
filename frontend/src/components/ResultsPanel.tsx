// ==========================================
// Results Panel Component
// Displays search results, reports, and selected documents
// ==========================================

import React from 'react'
import * as api from '../services/api'
import { useToasts, useModals } from '../context/SimulationContext'
import type { Report, DocMetadata } from '../types'

interface ResultsPanelProps {
  results: any[]
  reports: Report[]
  selectedDocs: DocMetadata[]
  setSelectedDocs: React.Dispatch<React.SetStateAction<DocMetadata[]>>
  loadingDoc: number | null
  setLoadingDoc: React.Dispatch<React.SetStateAction<number | null>>
  jobId?: string
  jobPct: number
}

export function ResultsPanel({
  results,
  reports,
  selectedDocs,
  setSelectedDocs,
  loadingDoc,
  setLoadingDoc,
  jobId,
  jobPct,
}: ResultsPanelProps) {
  const { addToast } = useToasts()
  const { autoScrollReport } = useModals()

  const handleLoadDocument = async (docId: number) => {
    try {
      setLoadingDoc(docId)
      const doc = await api.fetchDocument(docId)
      setSelectedDocs((prev) =>
        prev.map((x: any) =>
          x.id === docId ? { ...x, full: doc?.content || '', embedding: doc?.embedding || [] } : x
        )
      )
    } catch (e: any) {
      addToast('Fetch doc failed: ' + (e?.message || e))
    } finally {
      setLoadingDoc(null)
    }
  }

  return (
    <>
      {jobId && (
        <div className="card">
          <div className="section-title">Shard progress (job {jobId})</div>
          <div className="progress">
            <div className="progress-bar" style={{ width: `${jobPct}%` }} />
          </div>
        </div>
      )}

      <div className="card">
        <div className="section-title">Results</div>
        <div className="results">
          {selectedDocs.map((m, idx) => (
            <div key={'sel' + idx} className="result-item">
              <div style={{ display: 'flex', gap: '8px', alignItems: 'baseline' }}>
                <span className="score">{Number(m?.score || 0).toFixed(3)}</span>
                <span>doc #{m?.id}</span>
                <button
                  className="button secondary"
                  style={{ marginLeft: 'auto' }}
                  onClick={() => m?.id != null && handleLoadDocument(m.id)}
                >
                  Load content & embedding
                </button>
              </div>
              <span className="snippet">{String(m?.snippet || '')}</span>
              {loadingDoc === m?.id && <div style={{ opacity: 0.7, fontStyle: 'italic' }}>loading…</div>}
              {m?.full && (
                <pre
                  style={{
                    whiteSpace: 'pre-wrap',
                    background: 'rgba(0,0,0,0.04)',
                    padding: '8px',
                    borderRadius: 4,
                    marginTop: 6,
                    maxHeight: 200,
                    overflow: 'auto',
                  }}
                >
                  {String(m.full)}
                </pre>
              )}
              {Array.isArray(m?.embedding) && m.embedding.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  <div style={{ fontWeight: 600, opacity: 0.85 }}>Embedding ({m.embedding.length} dims)</div>
                  <div style={{ fontFamily: 'monospace', fontSize: 12, maxHeight: 120, overflow: 'auto' }}>
                    {String(
                      m.embedding
                        .slice(0, 64)
                        .map((v: number) => Number(v).toFixed(3))
                    ).replace(/,/g, ', ')}
                    {m.embedding.length > 64 ? ' …' : ''}
                  </div>
                </div>
              )}
            </div>
          ))}
          {results.map((it, idx) => (
            <div key={idx} className="result-item">
              <span className="score">{Number(it.relevance_score || 0).toFixed(3)}</span>
              <span>{it.metadata?.file_path || 'chunk'}</span>
              <span className="snippet">{String(it.content || '').slice(0, 180)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="section-title">Step Report</div>
        {reports.length === 0 ? (
          <div className="results" style={{ padding: 10 }}>
            No report yet.
          </div>
        ) : (
          <div className="results">
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', padding: '8px 12px' }}>
              <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <input type="checkbox" checked={autoScrollReport} readOnly /> Auto-scroll latest
              </label>
            </div>
            {(() => {
              const latest = reports[reports.length - 1]
              const items = Array.isArray(latest?.data?.items) ? latest.data.items : []
              return (
                <div>
                  <div style={{ padding: '8px 12px', fontWeight: 700 }}>
                    Step {latest.step} • Items {items.length}
                  </div>
                  {items.slice(0, 20).map((it: any, idx: number) => (
                    <div key={'rep' + idx} className="result-item">
                      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                        <span className="score">{Number(it?.embedding_score || 0).toFixed(3)}</span>
                        <span style={{ opacity: 0.85 }}>{String(it?.file_path || 'file')}</span>
                        <span style={{ opacity: 0.6 }}>
                          lines {Array.isArray(it?.line_range) ? it.line_range.join('-') : ''}
                        </span>
                      </div>
                      <div className="snippet">{String(it?.code_purpose || '')}</div>
                      {it?.relevance_to_query && <div className="snippet">why: {String(it.relevance_to_query)}</div>}
                    </div>
                  ))}
                  <div style={{ padding: 10 }}>
                    <button
                      className="button secondary"
                      onClick={() => {
                        try {
                          const blob = new Blob([JSON.stringify(latest.data, null, 2)], {
                            type: 'application/json',
                          })
                          const a = document.createElement('a')
                          a.href = URL.createObjectURL(blob)
                          a.download = `report_step_${latest.step}.json`
                          a.click()
                        } catch (e) {
                          /* noop */
                        }
                      }}
                    >
                      Save latest JSON
                    </button>
                  </div>
                </div>
              )
            })()}
          </div>
        )}
      </div>
    </>
  )
}
