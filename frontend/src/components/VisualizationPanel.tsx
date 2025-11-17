// ==========================================
// Visualization Panel Component
// 3D pheromone network visualization
// ==========================================

import React, { useMemo } from 'react'
import Plot from 'react-plotly.js'
import type { Snapshot, ThemeMode, DocMetadata } from '../types'

interface VisualizationPanelProps {
  snap: Snapshot | null
  dims: number
  theme: ThemeMode
  onDocClick: (doc: DocMetadata) => void
}

export function VisualizationPanel({ snap, dims, theme, onDocClick }: VisualizationPanelProps) {
  const fig = useMemo(() => {
    if (!snap) return { data: [], layout: { title: 'Waiting for data...' } }

    const data: any[] = []
    const edges = snap.edges || []
    const docs = snap.documents || { xy: [], relevance: [], meta: [] as any[] }
    const agents = snap.agents || { xy: [] }

    const prefersDark =
      theme === 'dark' ||
      (theme === 'system' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
    const paper = prefersDark ? '#0f1115' : '#f8f9fb'
    const plot = prefersDark ? '#11141a' : '#ffffff'
    const grid = prefersDark ? '#2a2f3a' : '#e5e7eb'
    const edgeColor = prefersDark ? 'rgba(59,130,246,0.35)' : 'rgba(0,150,0,0.25)'
    const docColor2D = prefersDark ? '#93c5fd' : 'rgba(0,0,200,0.7)'
    const agentColor2D = prefersDark ? 'rgba(248,113,113,0.85)' : 'rgba(200,0,0,0.7)'

    if (dims === 3) {
      for (const e of edges) {
        const z0 = 'z0' in e ? e.z0 : 0,
          z1 = 'z1' in e ? e.z1 : 0
        data.push({
          x: [e.x0, e.x1],
          y: [e.y0, e.y1],
          z: [z0, z1],
          mode: 'lines',
          type: 'scatter3d',
          line: { width: Math.max(1, (e.s || 0) * 3), color: edgeColor },
        })
      }
      if (docs.xy.length) {
        const xs = docs.xy.map((p) => p[0]),
          ys = docs.xy.map((p) => p[1]),
          zs = docs.xy.map((p) => p[2] || 0)
        const sizes = (docs.relevance || []).map((r) => 4 + 10 * r)
        const text = (docs.meta || []).map(
          (m: any, i: number) =>
            `id=${m?.id ?? i}<br>score=${(m?.score ?? 0).toFixed(3)}<br>visits=${m?.visits ?? 0}<br>${String(
              m?.snippet || ''
            ).replace(/</g, '&lt;')}`
        )
        data.push({
          x: xs,
          y: ys,
          z: zs,
          mode: 'markers',
          type: 'scatter3d',
          marker: {
            size: sizes,
            color: docs.relevance || [],
            colorscale: prefersDark ? 'Cividis' : 'Viridis',
            opacity: 0.9,
          },
          name: 'docs',
          text,
          hovertemplate: '%{text}<extra></extra>',
          customdata: docs.meta,
        })
      }
      if (agents.xy.length) {
        const xa = agents.xy.map((p) => p[0]),
          ya = agents.xy.map((p) => p[1]),
          za = agents.xy.map((p) => p[2] || 0)
        data.push({
          x: xa,
          y: ya,
          z: za,
          mode: 'markers',
          type: 'scatter3d',
          marker: { size: 3.5, color: agentColor2D },
          name: 'agents',
        })
      }
      return {
        data,
        layout: {
          height: 640,
          scene: {
            xaxis: { title: 'x', gridcolor: grid },
            yaxis: { title: 'y', gridcolor: grid },
            zaxis: { title: 'z', gridcolor: grid },
          },
          paper_bgcolor: paper,
          font: { color: prefersDark ? '#e5e7eb' : '#111827' },
          margin: { l: 0, r: 0, t: 30, b: 0 },
        },
      }
    } else {
      for (const e of edges)
        data.push({
          x: [e.x0, e.x1],
          y: [e.y0, e.y1],
          mode: 'lines',
          type: 'scatter',
          line: { width: Math.max(1, (e.s || 0) * 3), color: edgeColor },
          hoverinfo: 'skip',
        })
      if (docs.xy.length) {
        const xs = docs.xy.map((p) => p[0]),
          ys = docs.xy.map((p) => p[1])
        const sizes = (docs.relevance || []).map((r) => 8 + 12 * r)
        const text = (docs.meta || []).map(
          (m: any, i: number) =>
            `id=${m?.id ?? i}<br>score=${(m?.score ?? 0).toFixed(3)}<br>visits=${m?.visits ?? 0}<br>${String(
              m?.snippet || ''
            ).replace(/</g, '&lt;')}`
        )
        data.push({
          x: xs,
          y: ys,
          mode: 'markers',
          type: 'scatter',
          marker: { size: sizes, color: docColor2D },
          name: 'docs',
          text,
          hovertemplate: '%{text}<extra></extra>',
          customdata: (docs as any).meta,
        })
      }
      if (agents.xy.length) {
        const xa = agents.xy.map((p) => p[0]),
          ya = agents.xy.map((p) => p[1])
        data.push({
          x: xa,
          y: ya,
          mode: 'markers',
          type: 'scatter',
          marker: { size: 3.5, color: agentColor2D },
          name: 'agents',
        })
      }
      return {
        data,
        layout: {
          height: 600,
          xaxis: { title: 'x', gridcolor: grid },
          yaxis: { title: 'y', gridcolor: grid },
          paper_bgcolor: paper,
          plot_bgcolor: plot,
          font: { color: prefersDark ? '#e5e7eb' : '#111827' },
          margin: { l: 10, r: 10, t: 30, b: 10 },
        },
      }
    }
  }, [snap, dims, theme])

  const handleClick = (e: any) => {
    try {
      const p = e?.points?.[0]
      if (!p) return
      if (p.data?.name !== 'docs') return
      const idx = p.pointIndex
      const m = Array.isArray(p.data?.customdata) ? p.data.customdata[idx] : undefined
      if (m) {
        onDocClick({ ...m })
      }
    } catch {}
  }

  return (
    <div className="card">
      <div className="section-title">Live pheromone network</div>
      <Plot
        data={fig.data as any}
        layout={fig.layout as any}
        style={{ width: '100%', height: '600px' }}
        onClick={handleClick}
      />
    </div>
  )
}
