// ==========================================
// Type Definitions for Fungus MCMP Frontend
// ==========================================

export type ThemeMode = 'light' | 'dark' | 'system'

export interface Snapshot {
  documents: {
    xy: number[][]
    relevance: number[]
    meta?: any[]
  }
  agents: {
    xy: number[][]
  }
  edges: Array<any>
}

export interface Collection {
  name: string
  point_count: number
  dimension: number
  is_active: boolean
}

export interface CorpusStatus {
  docs: number
  files: number
  agents: number
}

export interface SearchResult {
  relevance_score?: number
  metadata?: {
    file_path?: string
  }
  content?: string
}

export interface ReportItem {
  embedding_score?: number
  file_path?: string
  line_range?: number[]
  code_purpose?: string
  relevance_to_query?: string
}

export interface Report {
  step: number
  data: {
    items?: ReportItem[]
  }
}

export interface EventLog {
  ts: number
  step?: number
  type: string
  text: string
}

export interface Settings {
  // Visualization
  viz_dims?: number
  min_trail_strength?: number
  max_edges?: number
  redraw_every?: number

  // Simulation
  num_agents?: number
  max_iterations?: number
  exploration_bonus?: number
  pheromone_decay?: number
  embed_batch_size?: number
  max_chunks_per_shard?: number

  // Query
  query?: string
  top_k?: number
  report_enabled?: boolean
  report_every?: number
  report_mode?: string
  mq_enabled?: boolean
  mq_count?: number
  judge_mode?: string

  // Corpus
  use_repo?: boolean
  root_folder?: string
  max_files?: number
  exclude_dirs?: string[] | string
  windows?: number[] | string
  chunk_workers?: number

  // LLM Settings
  llm_provider?: string

  // Ollama
  ollama_model?: string
  ollama_host?: string
  ollama_system?: string
  ollama_num_gpu?: number
  ollama_num_thread?: number
  ollama_num_batch?: number

  // OpenAI
  openai_model?: string
  openai_base_url?: string
  openai_temperature?: number

  // Google
  google_model?: string
  google_base_url?: string
  google_temperature?: number

  // Grok
  grok_model?: string
  grok_base_url?: string
  grok_temperature?: number

  // Legacy support
  max_steps?: number
}

export interface WebSocketMessage {
  type: string
  data?: any
  step?: number
  message?: string
  job_id?: string
  percent?: number
  added?: string[]
  pool_size?: number
}

export interface DocMetadata {
  id?: number
  score?: number
  visits?: number
  snippet?: string
  full?: string
  embedding?: number[]
}
