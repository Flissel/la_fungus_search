// ==========================================
// Simulation State Management Hook
// Manages simulation-related state
// ==========================================

import { useState, useCallback } from 'react'
import type { Snapshot, EventLog, Report, CorpusStatus, DocMetadata } from '../types'

export function useSimulationState() {
  const [snap, setSnap] = useState<Snapshot | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [results, setResults] = useState<any[]>([])
  const [reports, setReports] = useState<Report[]>([])
  const [events, setEvents] = useState<EventLog[]>([])
  const [selectedDocs, setSelectedDocs] = useState<DocMetadata[]>([])
  const [corpusStatus, setCorpusStatus] = useState<CorpusStatus | null>(null)
  const [jobId, setJobId] = useState<string | undefined>()
  const [jobPct, setJobPct] = useState<number>(0)
  const [loadingDoc, setLoadingDoc] = useState<number | null>(null)
  const [summaryJson, setSummaryJson] = useState<any | null>(null)

  // Metrics state
  const [mSteps, setMSteps] = useState<number[]>([])
  const [mAvg, setMAvg] = useState<number[]>([])
  const [mMax, setMMax] = useState<number[]>([])
  const [mTrails, setMTrails] = useState<number[]>([])
  const [mResults, setMResults] = useState<number[]>([])

  const addLog = useCallback((message: string) => {
    setLogs((prev) => {
      const last = prev[prev.length - 1]
      return last === message ? prev : [...prev.slice(-300), message]
    })
  }, [])

  const addEvent = useCallback((event: EventLog) => {
    setEvents((prev) => [...prev.slice(-99), event])
  }, [])

  const addReport = useCallback((step: number, data: any) => {
    setReports((prev) => {
      const next = [...prev, { step, data }]
      return next.slice(-50)
    })
    const itemCount = Array.isArray(data?.items) ? data.items.length : 0
    addLog(`report: step ${step} items=${itemCount}`)
    addEvent({
      ts: Date.now(),
      step,
      type: 'report',
      text: `items=${itemCount}`,
    })
  }, [addLog, addEvent])

  const addResults = useCallback((step: number, data: any[], type: string) => {
    setResults(data)
    const count = data.length
    addEvent({
      ts: Date.now(),
      step,
      type,
      text: `count=${count}`,
    })
    setMSteps((prev) => [...prev, step].slice(-200))
    setMResults((prev) => [...prev, count].slice(-200))
  }, [addEvent])

  const addSeedQueries = useCallback((added: string[], poolSize: number) => {
    const msg =
      `seed: +${added.length} → pool=${poolSize}\n` +
      added.map((q: string) => `- ${q}`).join('\n')
    addLog(msg)
    addEvent({
      ts: Date.now(),
      type: 'seed',
      text: `added=${added.length} pool=${poolSize}`,
    })
  }, [addLog, addEvent])

  const updateMetrics = useCallback((data: any) => {
    const step = Number(data.step || 0)
    const avg = Number(data.avg_rel || 0)
    const mx = Number(data.max_rel || 0)
    const tr = Number(data.trails || 0)
    const docs = Number(data.docs || 0)
    const files = Number(data.files || 0)
    const agents = Number(data.agents || 0)

    setMSteps((prev) => [...prev, step].slice(-200))
    setMAvg((prev) => [...prev, avg].slice(-200))
    setMMax((prev) => [...prev, mx].slice(-200))
    setMTrails((prev) => [...prev, tr].slice(-200))

    if (docs > 0 || agents > 0) {
      setCorpusStatus({ docs, files, agents })
    }

    addEvent({
      ts: Date.now(),
      step,
      type: 'metrics',
      text: `avg=${avg.toFixed(4)} trails=${tr}`,
    })

    addLog(`metrics: ${JSON.stringify(data)}`)
  }, [addEvent, addLog])

  const resetState = useCallback(() => {
    setSnap(null)
    setResults([])
    setReports([])
    setLogs([])
    setSelectedDocs([])
    setJobId(undefined)
    setJobPct(0)
  }, [])

  return {
    // State
    snap,
    logs,
    results,
    reports,
    events,
    selectedDocs,
    corpusStatus,
    jobId,
    jobPct,
    loadingDoc,
    summaryJson,
    mSteps,
    mAvg,
    mMax,
    mTrails,
    mResults,

    // Setters
    setSnap,
    setResults,
    setSelectedDocs,
    setJobId,
    setJobPct,
    setLoadingDoc,
    setSummaryJson,

    // Handlers
    addLog,
    addEvent,
    addReport,
    addResults,
    addSeedQueries,
    updateMetrics,
    resetState,
  }
}
