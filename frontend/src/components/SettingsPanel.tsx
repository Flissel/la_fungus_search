// ==========================================
// Settings Panel Component
// Configuration UI for simulation parameters
// ==========================================

import React from 'react'
import * as api from '../services/api'
import { useToasts, useCollections } from '../context/SimulationContext'
import type { ThemeMode, Settings, Collection } from '../types'

interface SettingsPanelProps {
  // Settings hook values
  dims: number
  minTrail: number
  setMinTrail: (v: number) => void
  maxEdges: number
  setMaxEdges: (v: number) => void
  redrawEvery: number
  setRedrawEvery: (v: number) => void
  numAgents: number
  setNumAgents: (v: number) => void
  maxIterations: number
  setMaxIterations: (v: number) => void
  explorationBonus: number
  setExplorationBonus: (v: number) => void
  pheromoneDecay: number
  setPheromoneDecay: (v: number) => void
  embedBatchSize: number
  setEmbedBatchSize: (v: number) => void
  maxChunksPerShard: number
  setMaxChunksPerShard: (v: number) => void
  query: string
  setQuery: (v: string) => void
  mode: string
  setMode: (v: string) => void
  reportEnabled: boolean
  setReportEnabled: (v: boolean) => void
  reportEvery: number
  setReportEvery: (v: number) => void
  topK: number
  setTopK: (v: number) => void
  mqEnabled: boolean
  setMqEnabled: (v: boolean) => void
  mqCount: number
  setMqCount: (v: number) => void
  judgeMode: string
  setJudgeMode: (v: string) => void
  llmProvider: string
  setLlmProvider: (v: string) => void
  ollamaModel: string
  setOllamaModel: (v: string) => void
  ollamaHost: string
  setOllamaHost: (v: string) => void
  ollamaSystem: string
  setOllamaSystem: (v: string) => void
  ollamaNumGpu: number
  setOllamaNumGpu: (v: number) => void
  ollamaNumThread: number
  setOllamaNumThread: (v: number) => void
  ollamaNumBatch: number
  setOllamaNumBatch: (v: number) => void
  openaiModel: string
  setOpenaiModel: (v: string) => void
  openaiBaseUrl: string
  setOpenaiBaseUrl: (v: string) => void
  openaiTemperature: number
  setOpenaiTemperature: (v: number) => void
  googleModel: string
  setGoogleModel: (v: string) => void
  googleBaseUrl: string
  setGoogleBaseUrl: (v: string) => void
  googleTemperature: number
  setGoogleTemperature: (v: number) => void
  grokModel: string
  setGrokModel: (v: string) => void
  grokBaseUrl: string
  setGrokBaseUrl: (v: string) => void
  grokTemperature: number
  setGrokTemperature: (v: number) => void
  useRepo: boolean
  setUseRepo: (v: boolean) => void
  rootFolder: string
  updateRootFolder: (v: string) => void
  rootFolderValid: boolean
  maxFiles: number
  setMaxFiles: (v: number) => void
  excludeDirs: string
  setExcludeDirs: (v: string) => void
  windows: string
  setWindows: (v: string) => void
  chunkWorkers: number
  setChunkWorkers: (v: number) => void
  theme: ThemeMode
  updateTheme: (v: ThemeMode) => void
  buildSettings: () => Settings
  corpusStatus: { docs: number; files: number; agents: number } | null
  onSendConfig: (dims: number) => void
  onBuildSummary: (summary: any) => void
}

export function SettingsPanel(props: SettingsPanelProps) {
  const { addToast } = useToasts()
  const { collections, setCollections, activeCollection, setActiveCollection } = useCollections()

  const handleApply = async () => {
    try {
      const settings = props.buildSettings()
      await api.updateSettings(settings)
      props.onSendConfig(props.dims)
    } catch (e: any) {
      addToast('Apply failed: ' + (e?.message || e))
    }
  }

  const handleBuildSummary = async () => {
    try {
      const summary = await api.mergeReports()
      props.onBuildSummary(summary)
      addToast('Summary built')
    } catch (e: any) {
      addToast('Merge reports failed: ' + (e?.message || e))
    }
  }

  const handleSwitchCollection = async (newCollection: string) => {
    try {
      await api.switchCollection(newCollection)
      setActiveCollection(newCollection)
      addToast(`Switched to collection '${newCollection}'`)
      // Refresh collections list to update point counts
      const res = await api.fetchCollections()
      if (res.status === 'ok') {
        setCollections(res.collections || [])
      }
    } catch (err: any) {
      addToast(`Switch failed: ${err?.response?.data?.message || err.message}`)
    }
  }

  return (
    <>
      <div className="row group">
        <div>
          <span className="label">Theme</span>
          <select
            className="select"
            value={props.theme}
            onChange={(e) => props.updateTheme(e.target.value as ThemeMode)}
          >
            <option value="system">System</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>
      </div>

      {props.corpusStatus && (
        <div
          className="group"
          style={{
            padding: '12px',
            background: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 8, fontSize: 14, color: 'var(--text)' }}>
            Codebase Index
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 13, color: 'var(--muted)' }}>Files:</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
              {props.corpusStatus.files.toLocaleString()}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 13, color: 'var(--muted)' }}>Chunks:</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
              {props.corpusStatus.docs.toLocaleString()}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 13, color: 'var(--muted)' }}>Agents:</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
              {props.corpusStatus.agents.toLocaleString()}
            </span>
          </div>
        </div>
      )}

      <details open>
        <summary style={{ cursor: 'pointer', fontWeight: 700 }}>Simulation</summary>
        <div className="group">
          <span className="label">Query</span>
          <input className="input" value={props.query} onChange={(e) => props.setQuery(e.target.value)} />
        </div>
        <div className="row group">
          <div>
            <span className="label" title="Overall objective for the analysis (e.g., understand architecture, find bugs)">
              Task Mode
            </span>
            <select className="select" value={props.mode} onChange={(e) => props.setMode(e.target.value)}>
              <option value="architecture">architecture</option>
              <option value="bugs">bugs</option>
              <option value="quality">quality</option>
              <option value="documentation">documentation</option>
              <option value="features">features</option>
              <option value="deep">deep</option>
              <option value="structure">structure</option>
              <option value="exploratory">exploratory</option>
              <option value="summary">summary</option>
              <option value="repair">repair</option>
            </select>
          </div>
          <div>
            <span className="label">Top K</span>
            <input
              className="number"
              type="number"
              step={1}
              value={props.topK}
              onChange={(e) => props.setTopK(parseInt(e.target.value) || 1)}
            />
          </div>
        </div>
        <div className="row group">
          <div>
            <label className="label">Per-step report</label>
            <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={props.reportEnabled}
                onChange={(e) => props.setReportEnabled(e.target.checked)}
              />{' '}
              Enable
            </label>
          </div>
          <div>
            <span className="label">Report every N steps</span>
            <input
              className="number"
              type="number"
              step={1}
              value={props.reportEvery}
              onChange={(e) => props.setReportEvery(parseInt(e.target.value) || 1)}
            />
          </div>
        </div>
      </details>

      <div className="row group">
        <div>
          <span className="label" title="Steering strategy for the MCMP simulation (how the judge evaluates chunks)">
            Judge Mode
          </span>
          <select
            className="select"
            value={props.judgeMode}
            onChange={(e) => props.setJudgeMode(e.target.value)}
          >
            <option value="steering">steering</option>
            <option value="focused">focused</option>
            <option value="exploratory">exploratory</option>
          </select>
        </div>
        <div>
          <label className="label">Multi-query (LLM assist)</label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input type="checkbox" checked={props.mqEnabled} onChange={(e) => props.setMqEnabled(e.target.checked)} />{' '}
            Enable
          </label>
        </div>
      </div>

      {props.mqEnabled && (
        <div className="group">
          <span className="label">Multi-query count</span>
          <input
            className="number"
            type="number"
            step={1}
            value={props.mqCount}
            onChange={(e) => props.setMqCount(parseInt(e.target.value) || 1)}
          />
        </div>
      )}

      <details>
        <summary style={{ cursor: 'pointer', fontWeight: 700 }}>Models</summary>
        <div className="row group">
          <div>
            <span className="label">LLM Provider</span>
            <select
              className="select"
              value={props.llmProvider}
              onChange={(e) => props.setLlmProvider(e.target.value)}
            >
              <option value="ollama">ollama</option>
              <option value="openai">openai</option>
              <option value="google">google</option>
              <option value="grok">grok</option>
            </select>
          </div>
        </div>

        {props.llmProvider === 'ollama' && (
          <>
            <div className="group">
              <span className="label">Ollama Model</span>
              <input
                className="input"
                list="ollama-models"
                value={props.ollamaModel}
                onChange={(e) => props.setOllamaModel(e.target.value)}
                placeholder="e.g., qwen2.5-coder:7b"
              />
              <datalist id="ollama-models">
                <option value="qwen2.5-coder:7b" />
                <option value="qwen2.5:7b" />
                <option value="llama3.1:8b" />
                <option value="mistral:7b" />
                <option value="codellama:7b" />
                <option value="deepseek-coder:6.7b" />
                <option value="phi3:mini-4k" />
              </datalist>
            </div>
            <div className="group">
              <span className="label">Ollama Host</span>
              <input
                className="input"
                value={props.ollamaHost}
                onChange={(e) => props.setOllamaHost(e.target.value)}
                placeholder="http://127.0.0.1:11434"
              />
            </div>
            <div className="group">
              <span className="label">Ollama System Prompt</span>
              <input
                className="input"
                value={props.ollamaSystem}
                onChange={(e) => props.setOllamaSystem(e.target.value)}
                placeholder="Optional system prompt"
              />
            </div>
            <div className="row group">
              <div>
                <span className="label">Ollama GPUs</span>
                <input
                  className="number"
                  type="number"
                  step={1}
                  value={props.ollamaNumGpu}
                  onChange={(e) => props.setOllamaNumGpu(parseInt(e.target.value) || 0)}
                />
              </div>
              <div>
                <span className="label">Ollama Threads</span>
                <input
                  className="number"
                  type="number"
                  step={1}
                  value={props.ollamaNumThread}
                  onChange={(e) => props.setOllamaNumThread(parseInt(e.target.value) || 0)}
                />
              </div>
            </div>
            <div className="group">
              <span className="label">Ollama Batch</span>
              <input
                className="number"
                type="number"
                step={1}
                value={props.ollamaNumBatch}
                onChange={(e) => props.setOllamaNumBatch(parseInt(e.target.value) || 0)}
              />
            </div>
          </>
        )}

        {props.llmProvider === 'openai' && (
          <>
            <div className="group">
              <span className="label">OpenAI Model</span>
              <input
                className="input"
                list="openai-models"
                value={props.openaiModel}
                onChange={(e) => props.setOpenaiModel(e.target.value)}
                placeholder="e.g., gpt-4o-mini"
              />
              <datalist id="openai-models">
                <option value="gpt-4o-mini" />
                <option value="gpt-4o" />
                <option value="gpt-4.1-mini" />
                <option value="gpt-4.1" />
                <option value="o3-mini" />
              </datalist>
            </div>
            <div className="group">
              <span className="label">OpenAI Base URL</span>
              <input
                className="input"
                value={props.openaiBaseUrl}
                onChange={(e) => props.setOpenaiBaseUrl(e.target.value)}
                placeholder="https://api.openai.com"
              />
            </div>
            <div className="group">
              <span className="label">OpenAI Temperature</span>
              <input
                className="number"
                type="number"
                step={0.1}
                value={props.openaiTemperature}
                onChange={(e) => props.setOpenaiTemperature(parseFloat(e.target.value) || 0)}
              />
            </div>
          </>
        )}

        {props.llmProvider === 'google' && (
          <>
            <div className="group">
              <span className="label">Google Model</span>
              <input
                className="input"
                list="google-models"
                value={props.googleModel}
                onChange={(e) => props.setGoogleModel(e.target.value)}
                placeholder="e.g., gemini-1.5-pro"
              />
              <datalist id="google-models">
                <option value="gemini-1.5-pro" />
                <option value="gemini-1.5-flash" />
                <option value="gemini-1.5-flash-8b" />
              </datalist>
            </div>
            <div className="group">
              <span className="label">Google Base URL</span>
              <input
                className="input"
                value={props.googleBaseUrl}
                onChange={(e) => props.setGoogleBaseUrl(e.target.value)}
                placeholder="https://generativelanguage.googleapis.com"
              />
            </div>
            <div className="group">
              <span className="label">Google Temperature</span>
              <input
                className="number"
                type="number"
                step={0.1}
                value={props.googleTemperature}
                onChange={(e) => props.setGoogleTemperature(parseFloat(e.target.value) || 0)}
              />
            </div>
          </>
        )}

        {props.llmProvider === 'grok' && (
          <>
            <div className="group">
              <span className="label">Grok Model</span>
              <input
                className="input"
                list="grok-models"
                value={props.grokModel}
                onChange={(e) => props.setGrokModel(e.target.value)}
                placeholder="e.g., grok-2-latest"
              />
              <datalist id="grok-models">
                <option value="grok-2-latest" />
                <option value="grok-2-mini" />
              </datalist>
            </div>
            <div className="group">
              <span className="label">Grok Base URL</span>
              <input
                className="input"
                value={props.grokBaseUrl}
                onChange={(e) => props.setGrokBaseUrl(e.target.value)}
                placeholder="https://api.x.ai"
              />
            </div>
            <div className="group">
              <span className="label">Grok Temperature</span>
              <input
                className="number"
                type="number"
                step={0.1}
                value={props.grokTemperature}
                onChange={(e) => props.setGrokTemperature(parseFloat(e.target.value) || 0)}
              />
            </div>
          </>
        )}
      </details>

      <details>
        <summary style={{ cursor: 'pointer', fontWeight: 700 }}>Chunking</summary>
        <div className="group">
          <span className="label">Windows (lines)</span>
          <input className="input" value={props.windows} onChange={(e) => props.setWindows(e.target.value)} />
        </div>
        <div className="group">
          <label>
            <input type="checkbox" checked={props.useRepo} onChange={(e) => props.setUseRepo(e.target.checked)} /> Use
            project src/ folder as codebase
          </label>
        </div>
        <div className="group">
          <span className="label">Root folder (used when src is off)</span>
          <input
            className="input"
            value={props.rootFolder}
            onChange={(e) => props.updateRootFolder(e.target.value)}
            placeholder="C:\Users\User\Desktop\EmbeddingGemma"
            style={
              !props.rootFolderValid ? { borderColor: '#ef4444', outline: '1px solid #ef4444' } : undefined
            }
          />
          {!props.rootFolderValid && (
            <div style={{ fontSize: 12, color: '#ef4444', marginTop: 4 }}>
              Invalid path format. Expected absolute path (e.g., C:\path or /path)
            </div>
          )}
        </div>
        <div className="row group">
          <div>
            <span className="label">Max files to index</span>
            <input
              className="number"
              type="number"
              step={50}
              value={props.maxFiles}
              onChange={(e) => props.setMaxFiles(parseInt(e.target.value) || 0)}
            />
          </div>
          <div>
            <span className="label">Chunk workers (threads)</span>
            <input
              className="number"
              type="number"
              step={1}
              value={props.chunkWorkers}
              onChange={(e) => props.setChunkWorkers(parseInt(e.target.value) || 1)}
            />
          </div>
        </div>
        <div className="group">
          <span className="label">Exclude folders</span>
          <input
            className="input"
            value={props.excludeDirs}
            onChange={(e) => props.setExcludeDirs(e.target.value)}
          />
        </div>
      </details>

      <details>
        <summary style={{ cursor: 'pointer', fontWeight: 700 }}>Simulation parameters</summary>
        <div className="row group">
          <div>
            <span className="label">Viz dims</span>
            <div className="select" style={{ padding: '8px 12px' }}>
              3D
            </div>
          </div>
          <div>
            <span className="label">Max edges</span>
            <input
              className="number"
              type="number"
              step={50}
              value={props.maxEdges}
              onChange={(e) => props.setMaxEdges(parseInt(e.target.value))}
            />
          </div>
        </div>
        <div className="row group">
          <div>
            <span className="label">Min trail strength</span>
            <input
              className="number"
              type="number"
              step={0.01}
              value={props.minTrail}
              onChange={(e) => props.setMinTrail(parseFloat(e.target.value))}
            />
          </div>
          <div>
            <span className="label">Redraw every N steps</span>
            <input
              className="number"
              type="number"
              step={1}
              value={props.redrawEvery}
              onChange={(e) => props.setRedrawEvery(parseInt(e.target.value) || 1)}
            />
          </div>
        </div>
        <div className="row group">
          <div>
            <span className="label">Agents</span>
            <input
              className="number"
              type="number"
              step={10}
              value={props.numAgents}
              onChange={(e) => props.setNumAgents(parseInt(e.target.value) || 1)}
            />
          </div>
          <div>
            <span className="label">Max iterations</span>
            <input
              className="number"
              type="number"
              step={10}
              value={props.maxIterations}
              onChange={(e) => props.setMaxIterations(parseInt(e.target.value) || 1)}
            />
          </div>
        </div>
        <div className="row group">
          <div>
            <span className="label">Exploration bonus</span>
            <input
              className="number"
              type="number"
              step={0.01}
              value={props.explorationBonus}
              onChange={(e) => props.setExplorationBonus(parseFloat(e.target.value))}
            />
          </div>
          <div>
            <span className="label">Pheromone decay</span>
            <input
              className="number"
              type="number"
              step={0.01}
              value={props.pheromoneDecay}
              onChange={(e) => props.setPheromoneDecay(parseFloat(e.target.value))}
            />
          </div>
        </div>
        <div className="row group">
          <div>
            <span className="label">Embedding batch size</span>
            <input
              className="number"
              type="number"
              step={16}
              value={props.embedBatchSize}
              onChange={(e) => props.setEmbedBatchSize(parseInt(e.target.value) || 16)}
            />
          </div>
          <div>
            <span className="label">Max chunks per shard</span>
            <input
              className="number"
              type="number"
              step={100}
              value={props.maxChunksPerShard}
              onChange={(e) => props.setMaxChunksPerShard(parseInt(e.target.value) || 0)}
            />
          </div>
        </div>
      </details>

      <div className="row group">
        <button className="button" onClick={handleApply}>
          Apply
        </button>
      </div>

      <div className="row group">
        <button className="button secondary" onClick={handleBuildSummary}>
          Build Summary
        </button>
        <button
          className="button secondary"
          onClick={() => {
            try {
              const obj = props.buildSettings()
              if (!obj) {
                alert('No summary available yet')
                return
              }
              const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' })
              const a = document.createElement('a')
              a.href = URL.createObjectURL(blob)
              a.download = 'summary.json'
              a.click()
            } catch (e) {
              /* noop */
            }
          }}
        >
          Download Summary
        </button>
      </div>

      <div className="group">
        <span className="label">Collection</span>
        <select
          className="select"
          value={activeCollection}
          onChange={(e) => handleSwitchCollection(e.target.value)}
        >
          {collections.length === 0 ? (
            <option value={activeCollection}>{activeCollection} (no collections loaded)</option>
          ) : (
            collections.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.point_count} points, {c.dimension}d) {c.is_active ? '✓' : ''}
              </option>
            ))
          )}
        </select>
      </div>
    </>
  )
}
