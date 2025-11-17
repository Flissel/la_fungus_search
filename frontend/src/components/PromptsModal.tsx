// ==========================================
// Prompts Modal Component
// Edit and manage prompts
// ==========================================

import React from 'react'
import * as api from '../services/api'
import { useToasts } from '../context/SimulationContext'

interface PromptsModalProps {
  show: boolean
  onClose: () => void
  promptModes: string[]
  promptDefaults: Record<string, string>
  promptOverrides: Record<string, string>
  setPromptOverrides: React.Dispatch<React.SetStateAction<Record<string, string>>>
  promptModeSel: string
  setPromptModeSel: React.Dispatch<React.SetStateAction<string>>
}

export function PromptsModal({
  show,
  onClose,
  promptModes,
  promptDefaults,
  promptOverrides,
  setPromptOverrides,
  promptModeSel,
  setPromptModeSel,
}: PromptsModalProps) {
  const { addToast } = useToasts()

  if (!show) return null

  const handleSave = async () => {
    try {
      await api.savePrompts(promptOverrides)
      addToast('Prompts saved')
    } catch (e: any) {
      addToast('Save prompts failed: ' + (e?.message || e))
    }
  }

  const handleLoadDefault = () => {
    setPromptOverrides((prev) => ({ ...prev, [promptModeSel]: promptDefaults[promptModeSel] || '' }))
  }

  const handleClearOverride = () => {
    setPromptOverrides((prev) => ({ ...prev, [promptModeSel]: '' }))
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>Prompt Editor</div>
          <button className="button secondary" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="modal-body" style={{ display: 'grid', gap: 12 }}>
          <div>
            <span className="label">Mode</span>
            <select className="select" value={promptModeSel} onChange={(e) => setPromptModeSel(e.target.value)}>
              {(promptModes || ['deep', 'structure', 'exploratory', 'summary', 'repair', 'steering']).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <div className="label">Instructions (override)</div>
            <textarea
              className="input"
              style={{ minHeight: 180 }}
              value={promptOverrides[promptModeSel] || ''}
              onChange={(e) =>
                setPromptOverrides((prev) => ({ ...prev, [promptModeSel]: e.target.value }))
              }
            />
          </div>
          <div>
            <div className="label">Default (read-only)</div>
            <textarea
              className="input"
              style={{ minHeight: 140, opacity: 0.7 }}
              value={promptDefaults[promptModeSel] || ''}
              readOnly
            />
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="button secondary" onClick={handleLoadDefault}>
              Load default into override
            </button>
            <button className="button" onClick={handleSave}>
              Save
            </button>
            <button className="button secondary" onClick={handleClearOverride}>
              Clear override
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
