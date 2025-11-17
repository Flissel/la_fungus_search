// ==========================================
// Search Panel Component
// Search controls and execution
// ==========================================

import React from 'react'
import * as api from '../services/api'
import { useToasts } from '../context/SimulationContext'

interface SearchPanelProps {
  query: string
  topK: number
  onResults: (results: any[]) => void
}

export function SearchPanel({ query, topK, onResults }: SearchPanelProps) {
  const { addToast } = useToasts()

  const handleSearch = async () => {
    try {
      const results = await api.search(query, topK)
      onResults(results)
    } catch (e: any) {
      addToast('Search failed: ' + (e?.message || e))
    }
  }

  const handleAnswer = async () => {
    try {
      const answer = await api.answer(query, topK)
      alert(answer || '')
    } catch (e: any) {
      addToast('Answer failed: ' + (e?.message || e))
    }
  }

  return (
    <div className="row group">
      <button className="button" onClick={handleSearch}>
        Search
      </button>
      <button className="button secondary" onClick={handleAnswer}>
        Answer
      </button>
    </div>
  )
}
