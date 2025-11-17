// ==========================================
// Metrics Panel Component
// Real-time metrics visualization
// ==========================================

import React from 'react'
import Plot from 'react-plotly.js'

interface MetricsPanelProps {
  mSteps: number[]
  mAvg: number[]
  mMax: number[]
  mTrails: number[]
  mResults: number[]
}

export function MetricsPanel({ mSteps, mAvg, mMax, mTrails, mResults }: MetricsPanelProps) {
  return (
    <div className="card">
      <div className="section-title">Metrics</div>
      <Plot
        data={
          [
            { x: mSteps, y: mAvg, type: 'scatter', mode: 'lines', name: 'avg_rel' },
            { x: mSteps, y: mMax, type: 'scatter', mode: 'lines', name: 'max_rel' },
            { x: mSteps, y: mTrails, type: 'scatter', mode: 'lines', name: 'trails', yaxis: 'y2' },
            { x: mSteps, y: mResults, type: 'scatter', mode: 'lines', name: 'results', yaxis: 'y2' },
          ] as any
        }
        layout={
          {
            height: 300,
            margin: { l: 30, r: 30, t: 10, b: 30 },
            yaxis: { title: 'rel' },
            yaxis2: { overlaying: 'y', side: 'right', title: 'count' },
          } as any
        }
        style={{ width: '100%', height: '300px' }}
      />
    </div>
  )
}
