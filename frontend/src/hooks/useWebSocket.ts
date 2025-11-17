// ==========================================
// WebSocket Custom Hook
// Manages WebSocket connection and events
// ==========================================

import { useEffect, useRef } from 'react'
import { API } from '../services/api'
import type { WebSocketMessage } from '../types'

interface WebSocketHandlers {
  onSnapshot?: (data: any) => void
  onReport?: (step: number, data: any) => void
  onResults?: (step: number, data: any[], type: string) => void
  onSeedQueries?: (added: string[], poolSize: number) => void
  onLog?: (message: string) => void
  onMetrics?: (data: any) => void
  onJobProgress?: (jobId: string, percent: number) => void
}

export function useWebSocket(handlers: WebSocketHandlers) {
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let stopped = false

    function connect() {
      if (stopped) return

      // Prefer relative /ws so Vite proxy can upgrade; fallback to backend URL
      const wsUrl = window.location.origin.replace('http', 'ws') + '/ws'
      const altUrl = API.replace('http', 'ws') + '/ws'
      const ws = new WebSocket(wsRef.current ? altUrl : wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        // Connected
      }

      ws.onmessage = (ev) => {
        try {
          const obj: WebSocketMessage = JSON.parse(ev.data)

          switch (obj.type) {
            case 'snapshot':
              handlers.onSnapshot?.(obj.data)
              break

            case 'report':
              if (handlers.onReport) {
                const step = Number(obj.step || 0)
                handlers.onReport(step, obj.data)
              }
              break

            case 'results':
            case 'results_stable':
              if (handlers.onResults && Array.isArray(obj.data)) {
                const step = Number(obj.step || 0)
                handlers.onResults(step, obj.data, obj.type)
              }
              break

            case 'seed_queries':
              if (handlers.onSeedQueries) {
                const added = Array.isArray(obj.added) ? obj.added : []
                const poolSize = Number(obj.pool_size || 0)
                handlers.onSeedQueries(added, poolSize)
              }
              break

            case 'log':
              if (handlers.onLog) {
                const message = String(obj.message || '')
                handlers.onLog(message)
              }
              break

            case 'metrics':
              handlers.onMetrics?.(obj.data || {})
              break

            case 'job_progress':
              if (handlers.onJobProgress) {
                handlers.onJobProgress(obj.job_id || '', obj.percent || 0)
              }
              break
          }
        } catch (err) {
          // Ignore parsing errors
        }
      }

      ws.onclose = () => {
        if (stopped) return
        setTimeout(connect, 1000) // Retry connection
      }

      ws.onerror = () => {
        try {
          ws.close()
        } catch {}
      }
    }

    connect()

    return () => {
      stopped = true
      try {
        wsRef.current?.close()
      } catch {}
    }
  }, []) // Handlers are not dependencies to avoid reconnection

  // Function to send config updates
  const sendConfig = (vizDims: number) => {
    try {
      wsRef.current?.send(JSON.stringify({ type: 'config', viz_dims: vizDims }))
    } catch {}
  }

  return { wsRef, sendConfig }
}
