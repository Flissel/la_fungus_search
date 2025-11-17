// ==========================================
// Control Panel Component
// Simulation control buttons
// ==========================================

import React, { useState } from 'react'
import * as api from '../services/api'
import { useToasts } from '../context/SimulationContext'
import type { Settings } from '../types'

interface ControlPanelProps {
  settings: Settings
  query: string
  numAgents: number
  setNumAgents: (n: number) => void
  onReset: () => void
  onStartJob: (jobId: string) => void
  onOpenCorpus: () => void
  onOpenPrompts: () => void
  onOpenCollections: () => void
}

export function ControlPanel({
  settings,
  query,
  numAgents,
  setNumAgents,
  onReset,
  onStartJob,
  onOpenCorpus,
  onOpenPrompts,
  onOpenCollections,
}: ControlPanelProps) {
  const { addToast } = useToasts()
  const [isBuilding, setIsBuilding] = useState(false)
  const [isIndexing, setIsIndexing] = useState(false)

  const handleStart = async () => {
    try {
      await api.startSimulation(settings)
      addToast('Simulation started')
    } catch (e: any) {
      addToast('Start failed: ' + (e?.message || e))
    }
  }

  const handleStop = async () => {
    try {
      await api.stopSimulation()
    } catch (e: any) {
      addToast('Stop failed: ' + (e?.message || e))
    }
  }

  const handlePause = async () => {
    try {
      await api.pauseSimulation()
      addToast('Paused')
    } catch (e: any) {
      addToast('Pause failed: ' + (e?.message || e))
    }
  }

  const handleResume = async () => {
    try {
      await api.resumeSimulation()
      addToast('Resumed')
    } catch (e: any) {
      addToast('Resume failed: ' + (e?.message || e))
    }
  }

  const handleReset = async () => {
    try {
      await api.resetSimulation()
      onReset()
      addToast('Reset complete')
    } catch (e: any) {
      addToast('Reset failed: ' + (e?.message || e))
    }
  }

  const handleAddAgents = async () => {
    const n = Number(prompt('Add how many agents?', '50') || '0')
    if (!n || n <= 0) return
    try {
      const r = await api.addAgents(n)
      addToast(`Added ${r?.added || n} agents (total=${r?.agents})`)
    } catch (e: any) {
      addToast('Add agents failed: ' + (e?.message || e))
    }
  }

  const handleResizeAgents = async () => {
    const c = Number(prompt('Resize to how many agents?', String(numAgents)) || String(numAgents))
    if (isNaN(c) || c < 0) return
    try {
      const r = await api.resizeAgents(c)
      setNumAgents(c)
      addToast(`Agents resized (total=${r?.agents})`)
    } catch (e: any) {
      addToast('Resize agents failed: ' + (e?.message || e))
    }
  }

  const handleStartJob = async () => {
    try {
      const r = await api.startJob(query)
      onStartJob(r.job_id)
      addToast('Job ' + r.job_id + ' started')
    } catch (e: any) {
      addToast('Job start failed: ' + (e?.message || e))
    }
  }

  const handleBuildCorpus = async () => {
    setIsBuilding(true)
    try {
      await api.reindexCorpus()
      addToast('Corpus built successfully')
    } catch (e: any) {
      addToast('Build corpus failed: ' + (e?.message || e))
    } finally {
      setIsBuilding(false)
    }
  }

  const handleIndexRepo = async () => {
    setIsIndexing(true)
    try {
      const result = await api.indexRepo()
      addToast(`Qdrant index built: ${result.files} files, ${result.points} points`)
    } catch (e: any) {
      addToast('Build Qdrant index failed: ' + (e?.message || e))
    } finally {
      setIsIndexing(false)
    }
  }

  return (
    <>
      <div className="row group">
        <button className="button" onClick={handleStart}>
          Start
        </button>
        <button className="button secondary" onClick={handleStop}>
          Stop
        </button>
        <button className="button secondary" onClick={handleReset}>
          Reset
        </button>
      </div>
      <div className="row group">
        <button className="button secondary" onClick={handlePause}>
          Pause
        </button>
        <button className="button secondary" onClick={handleResume}>
          Resume
        </button>
      </div>
      <div className="row group">
        <button className="button secondary" onClick={handleAddAgents}>
          Add Agents
        </button>
        <button className="button secondary" onClick={handleResizeAgents}>
          Resize Agents
        </button>
      </div>
      <div className="row group">
        <button className="button secondary" onClick={handleBuildCorpus} disabled={isBuilding}>
          {isBuilding ? 'Building...' : 'Build Corpus'}
        </button>
        <button className="button secondary" onClick={handleIndexRepo} disabled={isIndexing}>
          {isIndexing ? 'Indexing...' : 'Build Qdrant Index'}
        </button>
        <button className="button secondary" onClick={onOpenCorpus}>
          Corpus
        </button>
        <button className="button secondary" onClick={handleStartJob}>
          Shard Run
        </button>
      </div>
      <div className="row group">
        <button className="button secondary" onClick={onOpenPrompts}>
          Prompts
        </button>
        <button className="button secondary" onClick={onOpenCollections}>
          Collections
        </button>
      </div>
    </>
  )
}
