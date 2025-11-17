// ==========================================
// Refactored App Component
// Main orchestration using hooks and components
// ==========================================

import React, { useEffect } from 'react'
import { SimulationProvider, useToasts, useModals, useCollections } from '../context/SimulationContext'
import { useWebSocket } from '../hooks/useWebSocket'
import { useSimulationState } from '../hooks/useSimulationState'
import { useSettings } from '../hooks/useSettings'
import * as api from '../services/api'

// Components
import { SettingsPanel } from '../components/SettingsPanel'
import { ControlPanel } from '../components/ControlPanel'
import { SearchPanel } from '../components/SearchPanel'
import { VisualizationPanel } from '../components/VisualizationPanel'
import { MetricsPanel } from '../components/MetricsPanel'
import { LogPanel } from '../components/LogPanel'
import { ResultsPanel } from '../components/ResultsPanel'
import { StatusBar } from '../components/StatusBar'
import { CorpusModal } from '../components/CorpusModal'
import { PromptsModal } from '../components/PromptsModal'
import { CollectionsModal } from '../components/CollectionsModal'

import './App.css'

function AppContent() {
  const { toasts, addToast } = useToasts()
  const { collections, setCollections, activeCollection, setActiveCollection } = useCollections()
  const {
    showCorpus,
    setShowCorpus,
    showPrompts,
    setShowPrompts,
    showCollections,
    setShowCollections,
    corpusFiles,
    setCorpusFiles,
    corpusPage,
    setCorpusPage,
    corpusTotal,
    setCorpusTotal,
    promptModes,
    setPromptModes,
    promptDefaults,
    setPromptDefaults,
    promptOverrides,
    setPromptOverrides,
    promptModeSel,
    setPromptModeSel,
  } = useModals()

  // Settings hook
  const settings = useSettings()

  // Simulation state hook
  const simulation = useSimulationState()

  // WebSocket hook
  const { sendConfig } = useWebSocket({
    onSnapshot: simulation.setSnap,
    onReport: simulation.addReport,
    onResults: simulation.addResults,
    onSeedQueries: (added, poolSize) => {
      simulation.addSeedQueries(added, poolSize)
      addToast(`Seeded ${added.length} follow-up queries`)
    },
    onLog: simulation.addLog,
    onMetrics: simulation.updateMetrics,
    onJobProgress: (jobId, percent) => {
      simulation.setJobId(jobId)
      simulation.setJobPct(percent)
    },
  })

  // Load collections on mount
  useEffect(() => {
    async function loadCollections() {
      try {
        const res = await api.fetchCollections()
        if (res.status === 'ok') {
          setCollections(res.collections || [])
          setActiveCollection(res.active_collection || 'codebase')
        }
      } catch (e) {
        // Silently fail on initial load
      }
    }
    loadCollections()
  }, [setCollections, setActiveCollection])

  // Force reset modal states on mount
  useEffect(() => {
    setShowCorpus(false)
    setShowPrompts(false)
    setShowCollections(false)
  }, [setShowCorpus, setShowPrompts, setShowCollections])

  // Escape key handler to close modals
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowCorpus(false)
        setShowPrompts(false)
        setShowCollections(false)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [setShowCorpus, setShowPrompts, setShowCollections])

  // Handlers
  const handleOpenCorpus = async (page = 1) => {
    try {
      const r = await api.fetchCorpusList(page, 500)
      setCorpusFiles(r.files || [])
      setCorpusPage(r.page || 1)
      setCorpusTotal(r.total || 0)
      setShowCorpus(true)
    } catch (e: any) {
      addToast('Corpus list failed: ' + (e?.message || e))
    }
  }

  const handleOpenPrompts = async () => {
    try {
      const r = await api.fetchPrompts()
      setPromptModes(r.modes || [])
      setPromptDefaults(r.defaults || {})
      setPromptOverrides(r.overrides || {})
      setShowPrompts(true)
    } catch (e: any) {
      addToast('Load prompts failed: ' + (e?.message || e))
    }
  }

  const handleDocClick = (doc: any) => {
    simulation.setSelectedDocs((prev) => [{ ...doc }, ...prev].slice(0, 20))
  }

  return (
    <div className="layout" data-theme={settings.theme}>
      <aside className="sidebar">
        <div className="title">Fungus (MCMP) Frontend</div>

        <SettingsPanel
          {...settings}
          corpusStatus={simulation.corpusStatus}
          onSendConfig={sendConfig}
          onBuildSummary={simulation.setSummaryJson}
        />

        <ControlPanel
          settings={settings.buildSettings()}
          query={settings.query}
          numAgents={settings.numAgents}
          setNumAgents={settings.setNumAgents}
          onReset={simulation.resetState}
          onStartJob={(jobId) => {
            simulation.setJobId(jobId)
            simulation.setJobPct(0)
          }}
          onOpenCorpus={() => handleOpenCorpus(1)}
          onOpenPrompts={handleOpenPrompts}
          onOpenCollections={() => setShowCollections(true)}
        />

        <SearchPanel
          query={settings.query}
          topK={settings.topK}
          onResults={simulation.setResults}
        />
      </aside>

      <main className="main">
        <VisualizationPanel
          snap={simulation.snap}
          dims={settings.dims}
          theme={settings.theme}
          onDocClick={handleDocClick}
        />

        <LogPanel logs={simulation.logs} events={simulation.events} />

        <MetricsPanel
          mSteps={simulation.mSteps}
          mAvg={simulation.mAvg}
          mMax={simulation.mMax}
          mTrails={simulation.mTrails}
          mResults={simulation.mResults}
        />

        <ResultsPanel
          results={simulation.results}
          reports={simulation.reports}
          selectedDocs={simulation.selectedDocs}
          setSelectedDocs={simulation.setSelectedDocs}
          loadingDoc={simulation.loadingDoc}
          setLoadingDoc={simulation.setLoadingDoc}
          jobId={simulation.jobId}
          jobPct={simulation.jobPct}
        />
      </main>

      <CorpusModal
        show={showCorpus}
        onClose={() => setShowCorpus(false)}
        corpusFiles={corpusFiles}
        corpusPage={corpusPage}
        corpusTotal={corpusTotal}
      />

      <PromptsModal
        show={showPrompts}
        onClose={() => setShowPrompts(false)}
        promptModes={promptModes}
        promptDefaults={promptDefaults}
        promptOverrides={promptOverrides}
        setPromptOverrides={setPromptOverrides}
        promptModeSel={promptModeSel}
        setPromptModeSel={setPromptModeSel}
      />

      <CollectionsModal
        show={showCollections}
        onClose={() => setShowCollections(false)}
        activeCollection={activeCollection}
      />

      <StatusBar toasts={toasts} showCorpus={showCorpus} showPrompts={showPrompts} />
    </div>
  )
}

export default function App() {
  return (
    <SimulationProvider>
      <AppContent />
    </SimulationProvider>
  )
}
