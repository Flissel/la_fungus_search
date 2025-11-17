// ==========================================
// Settings Management Hook
// Manages application settings state
// ==========================================

import { useState, useEffect, useCallback } from 'react'
import { fetchSettings as fetchSettingsAPI } from '../services/api'
import type { ThemeMode } from '../types'

export function useSettings() {
  // Visualization
  const [dims] = useState<number>(3) // Force 3D only
  const [minTrail, setMinTrail] = useState<number>(0.05)
  const [maxEdges, setMaxEdges] = useState<number>(600)
  const [redrawEvery, setRedrawEvery] = useState<number>(2)

  // Simulation
  const [numAgents, setNumAgents] = useState<number>(200)
  const [maxIterations, setMaxIterations] = useState<number>(60)
  const [explorationBonus, setExplorationBonus] = useState<number>(0.1)
  const [pheromoneDecay, setPheromoneDecay] = useState<number>(0.95)
  const [embedBatchSize, setEmbedBatchSize] = useState<number>(64)
  const [maxChunksPerShard, setMaxChunksPerShard] = useState<number>(0)

  // Query
  const [query, setQuery] = useState<string>('Explain the architecture.')
  const [mode, setMode] = useState<string>('deep')
  const [reportEnabled, setReportEnabled] = useState<boolean>(false)
  const [reportEvery, setReportEvery] = useState<number>(5)
  const [topK, setTopK] = useState<number>(5)
  const [mqEnabled, setMqEnabled] = useState<boolean>(false)
  const [mqCount, setMqCount] = useState<number>(5)
  const [judgeMode, setJudgeMode] = useState<string>('steering')

  // LLM Settings
  const [llmProvider, setLlmProvider] = useState<string>('ollama')

  // Ollama
  const [ollamaModel, setOllamaModel] = useState<string>('')
  const [ollamaHost, setOllamaHost] = useState<string>('')
  const [ollamaSystem, setOllamaSystem] = useState<string>('')
  const [ollamaNumGpu, setOllamaNumGpu] = useState<number>(0)
  const [ollamaNumThread, setOllamaNumThread] = useState<number>(0)
  const [ollamaNumBatch, setOllamaNumBatch] = useState<number>(0)

  // OpenAI
  const [openaiModel, setOpenaiModel] = useState<string>('')
  const [openaiBaseUrl, setOpenaiBaseUrl] = useState<string>('')
  const [openaiTemperature, setOpenaiTemperature] = useState<number>(0)

  // Google
  const [googleModel, setGoogleModel] = useState<string>('')
  const [googleBaseUrl, setGoogleBaseUrl] = useState<string>('')
  const [googleTemperature, setGoogleTemperature] = useState<number>(0)

  // Grok
  const [grokModel, setGrokModel] = useState<string>('')
  const [grokBaseUrl, setGrokBaseUrl] = useState<string>('')
  const [grokTemperature, setGrokTemperature] = useState<number>(0)

  // Corpus
  const [useRepo, setUseRepo] = useState<boolean>(true)
  const [rootFolder, setRootFolder] = useState<string>('')
  const [rootFolderValid, setRootFolderValid] = useState<boolean>(true)
  const [maxFiles, setMaxFiles] = useState<number>(1000)
  const [excludeDirs, setExcludeDirs] = useState<string>('.venv,node_modules,.git,external')
  const [windows, setWindows] = useState<string>('16000')
  const [chunkWorkers, setChunkWorkers] = useState<number>(32)
  const [expansionLines, setExpansionLines] = useState<number>(50)

  // Theme
  const [theme, setTheme] = useState<ThemeMode>(() => {
    try {
      return (localStorage.getItem('eg_theme') as ThemeMode) || 'system'
    } catch {
      return 'system'
    }
  })

  // Load settings from backend on mount
  useEffect(() => {
    async function loadSettings() {
      try {
        const s = await fetchSettingsAPI()

        // Load corpus settings
        if (s.root_folder !== undefined) setRootFolder(s.root_folder)
        if (s.use_repo !== undefined) setUseRepo(s.use_repo)
        if (s.max_files !== undefined) setMaxFiles(s.max_files)
        if (s.exclude_dirs !== undefined) setExcludeDirs(String(s.exclude_dirs))
        if (s.windows !== undefined) setWindows(String(s.windows))
        if (s.expansion_lines !== undefined) setExpansionLines(s.expansion_lines)

        // Load simulation settings
        if (s.query !== undefined) setQuery(s.query)
        if (s.num_agents !== undefined) setNumAgents(s.num_agents)
        if (s.max_steps !== undefined) setMaxIterations(s.max_steps)
        if (s.top_k !== undefined) setTopK(s.top_k)
        if (s.report_mode !== undefined) setMode(s.report_mode)
        if (s.judge_mode !== undefined) setJudgeMode(s.judge_mode)

        // Load LLM settings
        if (s.ollama_model !== undefined) setOllamaModel(s.ollama_model)
        if (s.ollama_host !== undefined) setOllamaHost(s.ollama_host)
        if (s.ollama_system !== undefined) setOllamaSystem(s.ollama_system)
        if (s.ollama_num_gpu !== undefined) setOllamaNumGpu(s.ollama_num_gpu)
        if (s.ollama_num_thread !== undefined) setOllamaNumThread(s.ollama_num_thread)
      } catch (e) {
        // Silently fail - use default values
      }
    }
    loadSettings()
  }, [])

  const validateRootFolder = useCallback((path: string): boolean => {
    if (!path) return true
    // Basic path validation - check if looks like a valid path
    // Windows: C:\path or \\network\path
    // Unix: /path or ./path or ../path
    return (
      /^[a-zA-Z]:\\/.test(path) || // Windows absolute
      /^\\\\/.test(path) || // UNC path
      /^\//.test(path) || // Unix absolute
      /^\.\.?\//.test(path) // Relative path
    )
  }, [])

  const updateRootFolder = useCallback(
    (path: string) => {
      setRootFolder(path)
      setRootFolderValid(validateRootFolder(path))
    },
    [validateRootFolder]
  )

  const updateTheme = useCallback((newTheme: ThemeMode) => {
    setTheme(newTheme)
    try {
      localStorage.setItem('eg_theme', newTheme)
    } catch {}
  }, [])

  // Build settings object for API calls
  const buildSettings = useCallback(() => {
    const extra: any = {}

    if ((judgeMode || '').trim()) extra.judge_mode = (judgeMode || '').trim()
    if ((ollamaModel || '').trim()) extra.ollama_model = (ollamaModel || '').trim()
    if ((ollamaHost || '').trim()) extra.ollama_host = (ollamaHost || '').trim()
    if ((ollamaSystem || '').trim()) extra.ollama_system = (ollamaSystem || '').trim()
    if (Number(ollamaNumGpu) > 0) extra.ollama_num_gpu = Number(ollamaNumGpu)
    if (Number(ollamaNumThread) > 0) extra.ollama_num_thread = Number(ollamaNumThread)
    if (Number(ollamaNumBatch) > 0) extra.ollama_num_batch = Number(ollamaNumBatch)
    if ((llmProvider || '').trim()) extra.llm_provider = (llmProvider || '').trim()
    if ((openaiModel || '').trim()) extra.openai_model = (openaiModel || '').trim()
    if ((openaiBaseUrl || '').trim()) extra.openai_base_url = (openaiBaseUrl || '').trim()
    if (Number.isFinite(openaiTemperature)) extra.openai_temperature = Number(openaiTemperature)
    if ((googleModel || '').trim()) extra.google_model = (googleModel || '').trim()
    if ((googleBaseUrl || '').trim()) extra.google_base_url = (googleBaseUrl || '').trim()
    if (Number.isFinite(googleTemperature)) extra.google_temperature = Number(googleTemperature)
    if ((grokModel || '').trim()) extra.grok_model = (grokModel || '').trim()
    if ((grokBaseUrl || '').trim()) extra.grok_base_url = (grokBaseUrl || '').trim()
    if (Number.isFinite(grokTemperature)) extra.grok_temperature = Number(grokTemperature)

    return {
      viz_dims: 3,
      min_trail_strength: minTrail,
      max_edges: maxEdges,
      redraw_every: redrawEvery,
      num_agents: numAgents,
      max_iterations: maxIterations,
      exploration_bonus: explorationBonus,
      pheromone_decay: pheromoneDecay,
      embed_batch_size: embedBatchSize,
      max_chunks_per_shard: maxChunksPerShard,
      use_repo: useRepo,
      root_folder: rootFolder,
      max_files: maxFiles,
      exclude_dirs: excludeDirs
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
      windows: windows
        .split(',')
        .map((s) => parseInt(s.trim()))
        .filter((n) => !isNaN(n)),
      chunk_workers: chunkWorkers,
      expansion_lines: expansionLines,
      top_k: topK,
      report_enabled: reportEnabled,
      report_every: reportEvery,
      report_mode: mode,
      mq_enabled: mqEnabled,
      mq_count: mqCount,
      ...extra,
    }
  }, [
    dims,
    minTrail,
    maxEdges,
    redrawEvery,
    numAgents,
    maxIterations,
    explorationBonus,
    pheromoneDecay,
    embedBatchSize,
    maxChunksPerShard,
    query,
    mode,
    reportEnabled,
    reportEvery,
    topK,
    mqEnabled,
    mqCount,
    judgeMode,
    ollamaModel,
    ollamaHost,
    ollamaSystem,
    ollamaNumGpu,
    ollamaNumThread,
    ollamaNumBatch,
    llmProvider,
    openaiModel,
    openaiBaseUrl,
    openaiTemperature,
    googleModel,
    googleBaseUrl,
    googleTemperature,
    grokModel,
    grokBaseUrl,
    grokTemperature,
    useRepo,
    rootFolder,
    maxFiles,
    excludeDirs,
    windows,
    chunkWorkers,
  ])

  return {
    // Visualization
    dims,
    minTrail,
    setMinTrail,
    maxEdges,
    setMaxEdges,
    redrawEvery,
    setRedrawEvery,

    // Simulation
    numAgents,
    setNumAgents,
    maxIterations,
    setMaxIterations,
    explorationBonus,
    setExplorationBonus,
    pheromoneDecay,
    setPheromoneDecay,
    embedBatchSize,
    setEmbedBatchSize,
    maxChunksPerShard,
    setMaxChunksPerShard,

    // Query
    query,
    setQuery,
    mode,
    setMode,
    reportEnabled,
    setReportEnabled,
    reportEvery,
    setReportEvery,
    topK,
    setTopK,
    mqEnabled,
    setMqEnabled,
    mqCount,
    setMqCount,
    judgeMode,
    setJudgeMode,

    // LLM
    llmProvider,
    setLlmProvider,
    ollamaModel,
    setOllamaModel,
    ollamaHost,
    setOllamaHost,
    ollamaSystem,
    setOllamaSystem,
    ollamaNumGpu,
    setOllamaNumGpu,
    ollamaNumThread,
    setOllamaNumThread,
    ollamaNumBatch,
    setOllamaNumBatch,
    openaiModel,
    setOpenaiModel,
    openaiBaseUrl,
    setOpenaiBaseUrl,
    openaiTemperature,
    setOpenaiTemperature,
    googleModel,
    setGoogleModel,
    googleBaseUrl,
    setGoogleBaseUrl,
    googleTemperature,
    setGoogleTemperature,
    grokModel,
    setGrokModel,
    grokBaseUrl,
    setGrokBaseUrl,
    grokTemperature,
    setGrokTemperature,

    // Corpus
    useRepo,
    setUseRepo,
    rootFolder,
    updateRootFolder,
    rootFolderValid,
    maxFiles,
    setMaxFiles,
    excludeDirs,
    setExcludeDirs,
    windows,
    setWindows,
    chunkWorkers,
    setChunkWorkers,

    // Theme
    theme,
    updateTheme,

    // Helpers
    buildSettings,
  }
}
