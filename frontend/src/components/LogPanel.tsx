// ==========================================
// Log Panel Component
// Displays log messages and events
// ==========================================

import React from 'react'
import type { EventLog } from '../types'

interface LogPanelProps {
  logs: string[]
  events: EventLog[]
}

export function LogPanel({ logs, events }: LogPanelProps) {
  return (
    <>
      <div className="card">
        <div className="section-title">Live log</div>
        <div className="log">{logs.join('\n')}</div>
      </div>

      <div className="card">
        <div className="section-title">Events</div>
        <div className="results">
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '120px 70px 120px 1fr',
              gap: 6,
              padding: '6px 10px',
              fontWeight: 600,
              opacity: 0.8,
            }}
          >
            <div>Time</div>
            <div>Step</div>
            <div>Type</div>
            <div>Text</div>
          </div>
          {events
            .slice()
            .reverse()
            .map((e, idx) => (
              <div
                key={'evt' + idx}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '120px 70px 120px 1fr',
                  gap: 6,
                  padding: '4px 10px',
                  borderTop: '1px solid rgba(255,255,255,0.08)',
                }}
              >
                <div>{new Date(e.ts).toLocaleTimeString()}</div>
                <div>{e.step ?? '-'}</div>
                <div>
                  <span style={{ padding: '2px 6px', borderRadius: 6, background: 'rgba(99,102,241,0.2)' }}>
                    {e.type}
                  </span>
                </div>
                <div style={{ opacity: 0.85 }}>{e.text}</div>
              </div>
            ))}
        </div>
      </div>
    </>
  )
}
