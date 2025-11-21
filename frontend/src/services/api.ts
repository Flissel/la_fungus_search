// ==========================================
// API Service Layer
// Centralizes all HTTP calls to backend
// ==========================================

import axios from 'axios'
import type { Settings, SearchResult } from '../types'

const API = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8011'

// ==========================================
// Settings API
// ==========================================

export async function fetchSettings(): Promise<Settings> {
  const res = await axios.get(`${API}/settings`)
  return res.data
}

export async function updateSettings(settings: Settings): Promise<void> {
  await axios.post(`${API}/config`, settings)
}

// ==========================================
// Simulation Control API
// ==========================================

export async function startSimulation(settings: Settings): Promise<void> {
  await axios.post(`${API}/start`, settings)
}

export async function stopSimulation(force: boolean = false): Promise<void> {
  await axios.post(`${API}/stop?force=${force}`)
}

export async function pauseSimulation(): Promise<void> {
  await axios.post(`${API}/pause`)
}

export async function resumeSimulation(): Promise<void> {
  await axios.post(`${API}/resume`)
}

export async function resetSimulation(): Promise<void> {
  await axios.post(`${API}/reset`)
}

// ==========================================
// Agent Management API
// ==========================================

export async function addAgents(count: number): Promise<{ added: number; agents: number }> {
  const res = await axios.post(`${API}/agents/add`, { n: count })
  return res.data
}

export async function resizeAgents(count: number): Promise<{ agents: number }> {
  const res = await axios.post(`${API}/agents/resize`, { count })
  return res.data
}

// ==========================================
// Search & Query API
// ==========================================

export async function search(query: string, topK: number): Promise<SearchResult[]> {
  const res = await axios.post(`${API}/search`, { query, top_k: topK })
  return res.data.results || []
}

export async function answer(query: string, topK: number): Promise<string> {
  const res = await axios.post(`${API}/answer`, { query, top_k: topK })
  return res.data.answer || ''
}

// ==========================================
// Corpus API
// ==========================================

export async function fetchCorpusList(
  page: number = 1,
  pageSize: number = 500
): Promise<{ files: string[]; page: number; total: number }> {
  const res = await axios.get(`${API}/corpus/list?page=${page}&page_size=${pageSize}`)
  return res.data
}

export async function fetchDocument(docId: number): Promise<{ content: string; embedding: number[] }> {
  const res = await axios.get(`${API}/doc/${docId}`)
  return res.data?.doc || { content: '', embedding: [] }
}

// ==========================================
// Collections API
// ==========================================

export async function fetchCollections(): Promise<{
  status: string
  collections: Array<{ name: string; point_count: number; dimension: number; is_active: boolean }>
  active_collection: string
}> {
  const res = await axios.get(`${API}/collections/list`)
  return res.data
}

export async function switchCollection(collection: string): Promise<void> {
  await axios.post(`${API}/collections/switch`, { collection })
}

export async function deleteCollection(collection: string): Promise<void> {
  await axios.delete(`${API}/collections/${collection}`)
}

export async function reindexCorpus(): Promise<any> {
  const res = await axios.post(`${API}/corpus/reindex`, { force: true })
  return res.data
}

export async function indexRepo(): Promise<any> {
  const res = await axios.post(`${API}/corpus/index_repo`, {})
  return res.data
}

// ==========================================
// Reports API
// ==========================================

export async function mergeReports(): Promise<any> {
  const res = await axios.post(`${API}/reports/merge`, {})
  return res.data?.summary || null
}

// ==========================================
// Jobs API
// ==========================================

export async function startJob(query: string): Promise<{ job_id: string }> {
  const res = await axios.post(`${API}/jobs/start`, { query })
  return res.data
}

// ==========================================
// Prompts API
// ==========================================

export async function fetchPrompts(): Promise<{
  modes: string[]
  defaults: Record<string, string>
  overrides: Record<string, string>
}> {
  const res = await axios.get(`${API}/prompts`)
  return {
    modes: res.data?.modes || [],
    defaults: res.data?.defaults || {},
    overrides: res.data?.overrides || {},
  }
}

export async function savePrompts(overrides: Record<string, string>): Promise<void> {
  await axios.post(`${API}/prompts/save`, { overrides })
}

export { API }
