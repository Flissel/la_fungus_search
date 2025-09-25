import React, { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import './App.css'

const API = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8011'

type Snapshot = {
  documents: { xy: number[][], relevance: number[], meta?: any[] },
  agents: { xy: number[][] },
  edges: Array<any>
}

export default function App() {
  const [dims, setDims] = useState<number>(2)
  const [minTrail, setMinTrail] = useState<number>(0.05)
  const [maxEdges, setMaxEdges] = useState<number>(600)
  const [redrawEvery, setRedrawEvery] = useState<number>(2)
  const [numAgents, setNumAgents] = useState<number>(200)
  const [maxIterations, setMaxIterations] = useState<number>(60)
  const [explorationBonus, setExplorationBonus] = useState<number>(0.10)
  const [pheromoneDecay, setPheromoneDecay] = useState<number>(0.95)
  const [embedBatchSize, setEmbedBatchSize] = useState<number>(64)
  const [maxChunksPerShard, setMaxChunksPerShard] = useState<number>(0)
  const [query, setQuery] = useState<string>('Explain the architecture.')
  const [mode, setMode] = useState<string>('deep')
  const [reportEnabled, setReportEnabled] = useState<boolean>(false)
  const [reportEvery, setReportEvery] = useState<number>(5)
  const [topK, setTopK] = useState<number>(5)
  const [judgeMode, setJudgeMode] = useState<string>('steering')
  const [ollamaModel, setOllamaModel] = useState<string>('')
  const [ollamaHost, setOllamaHost] = useState<string>('')
  const [ollamaSystem, setOllamaSystem] = useState<string>('')
  const [ollamaNumGpu, setOllamaNumGpu] = useState<number>(0)
  const [ollamaNumThread, setOllamaNumThread] = useState<number>(0)
  const [ollamaNumBatch, setOllamaNumBatch] = useState<number>(0)
  const [llmProvider, setLlmProvider] = useState<string>('ollama')
  const [openaiModel, setOpenaiModel] = useState<string>('')
  const [openaiBaseUrl, setOpenaiBaseUrl] = useState<string>('')
  const [openaiTemperature, setOpenaiTemperature] = useState<number>(0)
  const [googleModel, setGoogleModel] = useState<string>('')
  const [googleBaseUrl, setGoogleBaseUrl] = useState<string>('')
  const [googleTemperature, setGoogleTemperature] = useState<number>(0)
  const [grokModel, setGrokModel] = useState<string>('')
  const [grokBaseUrl, setGrokBaseUrl] = useState<string>('')
  const [grokTemperature, setGrokTemperature] = useState<number>(0)
  const [useRepo, setUseRepo] = useState<boolean>(true)
  const [rootFolder, setRootFolder] = useState<string>('')
  const [maxFiles, setMaxFiles] = useState<number>(1000)
  const [excludeDirs, setExcludeDirs] = useState<string>('.venv,node_modules,.git,external')
  const [windows, setWindows] = useState<string>('1000,2000,4000')
  const [chunkWorkers, setChunkWorkers] = useState<number>(32)
  const [snap, setSnap] = useState<Snapshot | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [results, setResults] = useState<any[]>([])
  const [reports, setReports] = useState<Array<{step:number, data:any}>>([])
  type ThemeMode = 'light'|'dark'|'system'
  const [theme, setTheme] = useState<ThemeMode>(() => {
    try { return (localStorage.getItem('eg_theme') as ThemeMode) || 'system' } catch { return 'system' }
  })
  const [showCorpus, setShowCorpus] = useState(false)
  const [corpusFiles, setCorpusFiles] = useState<string[]>([])
  const [corpusPage, setCorpusPage] = useState(1)
  const [corpusTotal, setCorpusTotal] = useState(0)
  const [toasts, setToasts] = useState<string[]>([])
  const [autoScrollReport, setAutoScrollReport] = useState<boolean>(true)
  const [jobId, setJobId] = useState<string|undefined>()
  const [jobPct, setJobPct] = useState<number>(0)
  const [selectedDocs, setSelectedDocs] = useState<any[]>([])
  const [loadingDoc, setLoadingDoc] = useState<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let stopped = false
    function connect() {
      if (stopped) return
      // Prefer relative /ws so Vite proxy can upgrade; fallback to backend URL
      const wsUrl = (window.location.origin.replace('http', 'ws') + '/ws')
      const altUrl = API.replace('http', 'ws') + '/ws'
      const ws = new WebSocket(wsRef.current ? altUrl : wsUrl)
      wsRef.current = ws
      ws.onopen = () => {/* connected */}
      ws.onmessage = (ev) => {
        try {
          const obj = JSON.parse(ev.data)
          if (obj.type === 'snapshot') setSnap(obj.data)
          else if (obj.type === 'report') {
            setReports(prev => {
              const next = [...prev, { step: Number(obj.step||0), data: obj.data }]
              return next.slice(-50)
            })
            setLogs(prev => [...prev.slice(-300), `report: step ${obj.step} items=${Array.isArray(obj?.data?.items)? obj.data.items.length : 0}`])
          }
          else if (obj.type === 'results' || obj.type === 'results_stable') {
            if (Array.isArray(obj.data)) setResults(obj.data)
          }
          else if (obj.type === 'log') setLogs(prev => {
            const msg = String(obj.message || '')
            const last = prev[prev.length-1]
            return last === msg ? prev : [...prev.slice(-300), msg]
          })
          else if (obj.type === 'metrics') setLogs(prev => {
            const msg = `metrics: ${JSON.stringify(obj.data)}`
            const last = prev[prev.length-1]
            return last === msg ? prev : [...prev.slice(-300), msg]
          })
          else if (obj.type === 'job_progress') { setJobId(obj.job_id); setJobPct(obj.percent||0) }
        } catch {}
      }
      ws.onclose = () => {
        if (stopped) return
        setTimeout(connect, 1000) // retry
      }
      ws.onerror = () => {
        try { ws.close() } catch {}
      }
    }
    connect()
    return () => { stopped = true; try { wsRef.current?.close() } catch {} }
  }, [])

  async function apply() {
    const extra:any = {}
    if ((judgeMode||'').trim()) extra.judge_mode = (judgeMode||'').trim()
    if ((ollamaModel||'').trim()) extra.ollama_model = (ollamaModel||'').trim()
    if ((ollamaHost||'').trim()) extra.ollama_host = (ollamaHost||'').trim()
    if ((ollamaSystem||'').trim()) extra.ollama_system = (ollamaSystem||'').trim()
    if (Number(ollamaNumGpu) > 0) extra.ollama_num_gpu = Number(ollamaNumGpu)
    if (Number(ollamaNumThread) > 0) extra.ollama_num_thread = Number(ollamaNumThread)
    if (Number(ollamaNumBatch) > 0) extra.ollama_num_batch = Number(ollamaNumBatch)
    if ((llmProvider||'').trim()) extra.llm_provider = (llmProvider||'').trim()
    if ((openaiModel||'').trim()) extra.openai_model = (openaiModel||'').trim()
    if ((openaiBaseUrl||'').trim()) extra.openai_base_url = (openaiBaseUrl||'').trim()
    if (Number.isFinite(openaiTemperature)) extra.openai_temperature = Number(openaiTemperature)
    if ((googleModel||'').trim()) extra.google_model = (googleModel||'').trim()
    if ((googleBaseUrl||'').trim()) extra.google_base_url = (googleBaseUrl||'').trim()
    if (Number.isFinite(googleTemperature)) extra.google_temperature = Number(googleTemperature)
    if ((grokModel||'').trim()) extra.grok_model = (grokModel||'').trim()
    if ((grokBaseUrl||'').trim()) extra.grok_base_url = (grokBaseUrl||'').trim()
    if (Number.isFinite(grokTemperature)) extra.grok_temperature = Number(grokTemperature)
    await axios.post(API + '/config', {
      viz_dims: dims,
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
      exclude_dirs: excludeDirs.split(',').map(s=>s.trim()).filter(Boolean),
      windows: windows.split(',').map(s=>parseInt(s.trim())).filter(n=>!isNaN(n)),
      chunk_workers: chunkWorkers,
      top_k: topK,
      report_enabled: reportEnabled,
      report_every: reportEvery,
      report_mode: mode,
      ...extra,
    })
    try { wsRef.current?.send(JSON.stringify({ type: 'config', viz_dims: dims })) } catch {}
  }

  async function start() {
    try {
      const extra:any = {}
      if ((judgeMode||'').trim()) extra.judge_mode = (judgeMode||'').trim()
      if ((ollamaModel||'').trim()) extra.ollama_model = (ollamaModel||'').trim()
      if ((ollamaHost||'').trim()) extra.ollama_host = (ollamaHost||'').trim()
      if ((ollamaSystem||'').trim()) extra.ollama_system = (ollamaSystem||'').trim()
      if (Number(ollamaNumGpu) > 0) extra.ollama_num_gpu = Number(ollamaNumGpu)
      if (Number(ollamaNumThread) > 0) extra.ollama_num_thread = Number(ollamaNumThread)
      if (Number(ollamaNumBatch) > 0) extra.ollama_num_batch = Number(ollamaNumBatch)
      if ((llmProvider||'').trim()) extra.llm_provider = (llmProvider||'').trim()
      if ((openaiModel||'').trim()) extra.openai_model = (openaiModel||'').trim()
      if ((openaiBaseUrl||'').trim()) extra.openai_base_url = (openaiBaseUrl||'').trim()
      if (Number.isFinite(openaiTemperature)) extra.openai_temperature = Number(openaiTemperature)
      if ((googleModel||'').trim()) extra.google_model = (googleModel||'').trim()
      if ((googleBaseUrl||'').trim()) extra.google_base_url = (googleBaseUrl||'').trim()
      if (Number.isFinite(googleTemperature)) extra.google_temperature = Number(googleTemperature)
      if ((grokModel||'').trim()) extra.grok_model = (grokModel||'').trim()
      if ((grokBaseUrl||'').trim()) extra.grok_base_url = (grokBaseUrl||'').trim()
      if (Number.isFinite(grokTemperature)) extra.grok_temperature = Number(grokTemperature)
      await axios.post(API + '/start', {
        query,
        viz_dims: dims,
        use_repo: useRepo,
        root_folder: rootFolder,
        max_files: maxFiles,
        exclude_dirs: excludeDirs.split(',').map(s=>s.trim()).filter(Boolean),
        windows: windows.split(',').map(s=>parseInt(s.trim())).filter(n=>!isNaN(n)),
        chunk_workers: chunkWorkers,
        min_trail_strength: minTrail,
        max_edges: maxEdges,
        redraw_every: redrawEvery,
        num_agents: numAgents,
        max_iterations: maxIterations,
        exploration_bonus: explorationBonus,
        pheromone_decay: pheromoneDecay,
        embed_batch_size: embedBatchSize,
        max_chunks_per_shard: maxChunksPerShard,
        top_k: topK,
        report_enabled: reportEnabled,
        report_every: reportEvery,
        report_mode: mode,
        ...extra,
      })
      setToasts(t => [...t, 'Simulation started'])
    } catch (e:any) {
      setToasts(t => [...t, 'Start failed: ' + (e?.message||e)])
    }
  }

  async function doSearch() {
    try {
      const r = await axios.post(API + '/search', { query, top_k: topK })
      setResults(r.data.results || [])
    } catch (e:any) {
      setToasts(t => [...t, 'Search failed: ' + (e?.message||e)])
    }
  }

  async function doAnswer() {
    try {
      const r = await axios.post(API + '/answer', { query, top_k: topK })
      alert(r.data.answer || '')
    } catch (e:any) {
      setToasts(t => [...t, 'Answer failed: ' + (e?.message||e)])
    }
  }

  async function openCorpus(page=1) {
    try {
      const r = await axios.get(API + `/corpus/list?page=${page}&page_size=500`)
      setCorpusFiles(r.data.files || [])
      setCorpusPage(r.data.page || 1)
      setCorpusTotal(r.data.total || 0)
      setShowCorpus(true)
    } catch (e:any) {
      setToasts(t => [...t, 'Corpus list failed: ' + (e?.message||e)])
    }
  }

  const fig = useMemo(() => {
    if (!snap) return { data: [], layout: { title: 'Waiting for data...' } }
    const data: any[] = []
    const edges = snap.edges || []
    const docs = snap.documents || { xy: [], relevance: [], meta: [] as any[] }
    const agents = snap.agents || { xy: [] }
    const prefersDark = (theme === 'dark') || (theme === 'system' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
    const paper = prefersDark ? '#0f1115' : '#f8f9fb'
    const plot = prefersDark ? '#11141a' : '#ffffff'
    const grid = prefersDark ? '#2a2f3a' : '#e5e7eb'
    const edgeColor = prefersDark ? 'rgba(59,130,246,0.35)' : 'rgba(0,150,0,0.25)'
    const docColor2D = prefersDark ? '#93c5fd' : 'rgba(0,0,200,0.7)'
    const agentColor2D = prefersDark ? 'rgba(248,113,113,0.85)' : 'rgba(200,0,0,0.7)'
    if (dims === 3) {
      for (const e of edges) {
        const z0 = ('z0' in e) ? e.z0 : 0, z1 = ('z1' in e) ? e.z1 : 0
        data.push({ x:[e.x0,e.x1], y:[e.y0,e.y1], z:[z0,z1], mode:'lines', type:'scatter3d', line:{width:Math.max(1, (e.s||0)*3), color: edgeColor} })
      }
      if (docs.xy.length) {
        const xs = docs.xy.map(p=>p[0]), ys = docs.xy.map(p=>p[1]), zs = docs.xy.map(p=>p[2]||0)
        const sizes = (docs.relevance||[]).map(r=>4+10*r)
        const text = (docs.meta||[]).map((m:any,i:number)=>`id=${m?.id ?? i}<br>score=${(m?.score??0).toFixed(3)}<br>visits=${m?.visits ?? 0}<br>${String(m?.snippet||'').replace(/</g,'&lt;')}`)
        data.push({ x:xs, y:ys, z:zs, mode:'markers', type:'scatter3d', marker:{ size:sizes, color:docs.relevance||[], colorscale: prefersDark ? 'Cividis' : 'Viridis', opacity:0.9 }, name:'docs', text, hovertemplate: '%{text}<extra></extra>', customdata: docs.meta })
      }
      if (agents.xy.length) {
        const xa = agents.xy.map(p=>p[0]), ya = agents.xy.map(p=>p[1]), za = agents.xy.map(p=>p[2]||0)
        data.push({ x:xa, y:ya, z:za, mode:'markers', type:'scatter3d', marker:{ size:3.5, color: agentColor2D }, name:'agents' })
      }
      return { data, layout: { height: 640, scene: { xaxis:{title:'x', gridcolor:grid}, yaxis:{title:'y', gridcolor:grid}, zaxis:{title:'z', gridcolor:grid} }, paper_bgcolor: paper, font:{color: prefersDark ? '#e5e7eb' : '#111827'}, margin:{l:0,r:0,t:30,b:0} } }
    } else {
      for (const e of edges) data.push({ x:[e.x0,e.x1], y:[e.y0,e.y1], mode:'lines', type:'scatter', line:{width:Math.max(1,(e.s||0)*3), color: edgeColor}, hoverinfo:'skip' })
      if (docs.xy.length) {
        const xs = docs.xy.map(p=>p[0]), ys = docs.xy.map(p=>p[1])
        const sizes = (docs.relevance||[]).map(r=>8+12*r)
        const text = (docs.meta||[]).map((m:any,i:number)=>`id=${m?.id ?? i}<br>score=${(m?.score??0).toFixed(3)}<br>visits=${m?.visits ?? 0}<br>${String(m?.snippet||'').replace(/</g,'&lt;')}`)
        data.push({ x:xs, y:ys, mode:'markers', type:'scatter', marker:{ size:sizes, color: docColor2D }, name:'docs', text, hovertemplate: '%{text}<extra></extra>', customdata: (docs as any).meta })
      }
      if (agents.xy.length) {
        const xa = agents.xy.map(p=>p[0]), ya = agents.xy.map(p=>p[1])
        data.push({ x:xa, y:ya, mode:'markers', type:'scatter', marker:{ size:3.5, color: agentColor2D }, name:'agents' })
      }
      return { data, layout: { height: 600, xaxis:{title:'x', gridcolor:grid}, yaxis:{title:'y', gridcolor:grid}, paper_bgcolor: paper, plot_bgcolor: plot, font:{color: prefersDark ? '#e5e7eb' : '#111827'}, margin:{l:10,r:10,t:30,b:10} } }
    }
  }, [snap, dims, theme])

  return (
    <div className='layout' data-theme={theme}>
      <aside className='sidebar'>
        <div className='title'>Fungus (MCMP) Frontend</div>
        <div className='row group'>
          <div>
            <span className='label'>Theme</span>
            <select className='select' value={theme} onChange={e=>{ const v=e.target.value as ThemeMode; setTheme(v); try{localStorage.setItem('eg_theme', v)}catch{} }}>
              <option value='system'>System</option>
              <option value='light'>Light</option>
              <option value='dark'>Dark</option>
            </select>
          </div>
        </div>
        <div className='group'>
          <span className='label'>Query</span>
          <input className='input' value={query} onChange={e=>setQuery(e.target.value)} />
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Mode</span>
            <select className='select' value={mode} onChange={e=>setMode(e.target.value)}>
              <option value='deep'>deep</option>
              <option value='structure'>structure</option>
              <option value='exploratory'>exploratory</option>
              <option value='summary'>summary</option>
              <option value='steering'>steering</option>
              <option value='similar'>similar</option>
              <option value='redundancy'>redundancy</option>
              <option value='repair'>repair</option>
            </select>
          </div>
          <div>
            <span className='label'>Top K</span>
            <input className='number' type='number' step={1} value={topK} onChange={e=>setTopK(parseInt(e.target.value)||1)} />
          </div>
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Judge Mode</span>
            <select className='select' value={judgeMode} onChange={e=>setJudgeMode(e.target.value)}>
              <option value='steering'>steering</option>
              <option value='deep'>deep</option>
              <option value='structure'>structure</option>
              <option value='exploratory'>exploratory</option>
              <option value='summary'>summary</option>
              <option value='repair'>repair</option>
            </select>
          </div>
        </div>
        <div className='row group'>
          <div>
            <span className='label'>LLM Provider</span>
            <select className='select' value={llmProvider} onChange={e=>setLlmProvider(e.target.value)}>
              <option value='ollama'>ollama</option>
              <option value='openai'>openai</option>
              <option value='google'>google</option>
              <option value='grok'>grok</option>
            </select>
          </div>
        </div>
        <div className='group'>
          <span className='label'>Ollama Model</span>
          <input className='input' value={ollamaModel} onChange={e=>setOllamaModel(e.target.value)} placeholder='e.g., qwen2.5-coder:7b' />
        </div>
        <div className='group'>
          <span className='label'>Ollama Host</span>
          <input className='input' value={ollamaHost} onChange={e=>setOllamaHost(e.target.value)} placeholder='http://127.0.0.1:11434' />
        </div>
        <div className='group'>
          <span className='label'>Ollama System Prompt</span>
          <input className='input' value={ollamaSystem} onChange={e=>setOllamaSystem(e.target.value)} placeholder='Optional system prompt' />
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Ollama GPUs</span>
            <input className='number' type='number' step={1} value={ollamaNumGpu} onChange={e=>setOllamaNumGpu(parseInt(e.target.value)||0)} />
          </div>
          <div>
            <span className='label'>Ollama Threads</span>
            <input className='number' type='number' step={1} value={ollamaNumThread} onChange={e=>setOllamaNumThread(parseInt(e.target.value)||0)} />
          </div>
        </div>
        <div className='group'>
          <span className='label'>Ollama Batch</span>
          <input className='number' type='number' step={1} value={ollamaNumBatch} onChange={e=>setOllamaNumBatch(parseInt(e.target.value)||0)} />
        </div>
        <div className='group'>
          <span className='label'>OpenAI Model</span>
          <input className='input' value={openaiModel} onChange={e=>setOpenaiModel(e.target.value)} placeholder='e.g., gpt-4o-mini' />
        </div>
        <div className='group'>
          <span className='label'>OpenAI Base URL</span>
          <input className='input' value={openaiBaseUrl} onChange={e=>setOpenaiBaseUrl(e.target.value)} placeholder='https://api.openai.com' />
        </div>
        <div className='group'>
          <span className='label'>OpenAI Temperature</span>
          <input className='number' type='number' step={0.1} value={openaiTemperature} onChange={e=>setOpenaiTemperature(parseFloat(e.target.value)||0)} />
        </div>
        {llmProvider==='google' && (
          <>
            <div className='group'>
              <span className='label'>Google Model</span>
              <input className='input' value={googleModel} onChange={e=>setGoogleModel(e.target.value)} placeholder='e.g., gemini-1.5-pro' />
            </div>
            <div className='group'>
              <span className='label'>Google Base URL</span>
              <input className='input' value={googleBaseUrl} onChange={e=>setGoogleBaseUrl(e.target.value)} placeholder='https://generativelanguage.googleapis.com' />
            </div>
            <div className='group'>
              <span className='label'>Google Temperature</span>
              <input className='number' type='number' step={0.1} value={googleTemperature} onChange={e=>setGoogleTemperature(parseFloat(e.target.value)||0)} />
            </div>
          </>
        )}
        {llmProvider==='grok' && (
          <>
            <div className='group'>
              <span className='label'>Grok Model</span>
              <input className='input' value={grokModel} onChange={e=>setGrokModel(e.target.value)} placeholder='e.g., grok-2-latest' />
            </div>
            <div className='group'>
              <span className='label'>Grok Base URL</span>
              <input className='input' value={grokBaseUrl} onChange={e=>setGrokBaseUrl(e.target.value)} placeholder='https://api.x.ai' />
            </div>
            <div className='group'>
              <span className='label'>Grok Temperature</span>
              <input className='number' type='number' step={0.1} value={grokTemperature} onChange={e=>setGrokTemperature(parseFloat(e.target.value)||0)} />
            </div>
          </>
        )}
        <div className='group'>
          <span className='label'>Windows (lines)</span>
          <input className='input' value={windows} onChange={e=>setWindows(e.target.value)} />
        </div>
        <div className='group'>
          <label><input type='checkbox' checked={useRepo} onChange={e=>setUseRepo(e.target.checked)} /> Use code space (src) as corpus</label>
        </div>
        <div className='group'>
          <span className='label'>Root folder (used when src is off)</span>
          <input className='input' value={rootFolder} onChange={e=>setRootFolder(e.target.value)} placeholder='C:\\Users\\User\\Desktop\\EmbeddingGemma' />
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Max files to index</span>
            <input className='number' type='number' step={50} value={maxFiles} onChange={e=>setMaxFiles(parseInt(e.target.value)||0)} />
          </div>
          <div>
            <span className='label'>Chunk workers (threads)</span>
            <input className='number' type='number' step={1} value={chunkWorkers} onChange={e=>setChunkWorkers(parseInt(e.target.value)||1)} />
          </div>
        </div>
        <div className='group'>
          <span className='label'>Exclude folders</span>
          <input className='input' value={excludeDirs} onChange={e=>setExcludeDirs(e.target.value)} />
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Viz dims</span>
            <select className='select' value={dims} onChange={e=>setDims(parseInt(e.target.value))}><option value={2}>2D</option><option value={3}>3D</option></select>
          </div>
          <div>
            <span className='label'>Max edges</span>
            <input className='number' type='number' step={50} value={maxEdges} onChange={e=>setMaxEdges(parseInt(e.target.value))} />
          </div>
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Min trail strength</span>
            <input className='number' type='number' step={0.01} value={minTrail} onChange={e=>setMinTrail(parseFloat(e.target.value))} />
          </div>
          <div>
            <span className='label'>Redraw every N steps</span>
            <input className='number' type='number' step={1} value={redrawEvery} onChange={e=>setRedrawEvery(parseInt(e.target.value)||1)} />
          </div>
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Agents</span>
            <input className='number' type='number' step={10} value={numAgents} onChange={e=>setNumAgents(parseInt(e.target.value)||1)} />
          </div>
          <div>
            <span className='label'>Max iterations</span>
            <input className='number' type='number' step={10} value={maxIterations} onChange={e=>setMaxIterations(parseInt(e.target.value)||1)} />
          </div>
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Exploration bonus</span>
            <input className='number' type='number' step={0.01} value={explorationBonus} onChange={e=>setExplorationBonus(parseFloat(e.target.value))} />
          </div>
          <div>
            <span className='label'>Pheromone decay</span>
            <input className='number' type='number' step={0.01} value={pheromoneDecay} onChange={e=>setPheromoneDecay(parseFloat(e.target.value))} />
          </div>
        </div>
        <div className='row group'>
          <div>
            <span className='label'>Embedding batch size</span>
            <input className='number' type='number' step={16} value={embedBatchSize} onChange={e=>setEmbedBatchSize(parseInt(e.target.value)||16)} />
          </div>
          <div>
            <span className='label'>Max chunks per shard</span>
            <input className='number' type='number' step={100} value={maxChunksPerShard} onChange={e=>setMaxChunksPerShard(parseInt(e.target.value)||0)} />
          </div>
        </div>
        <div className='row group'>
          <button className='button' onClick={apply}>Apply</button>
        </div>
        <div className='row group'>
          <button className='button' onClick={start}>Start</button>
          <button className='button secondary' onClick={async ()=>{ await axios.post(API+'/stop'); }}>Stop</button>
          <button className='button secondary' onClick={async ()=>{ try{ await axios.post(API+'/reset'); setSnap(null); setResults([]); setReports([]); setLogs([]); setSelectedDocs([]); setJobId(undefined); setJobPct(0); setToasts(t=>[...t,'Reset complete']) } catch(e:any){ setToasts(t=>[...t,'Reset failed: '+(e?.message||e)]) } }}>Reset</button>
        </div>
        <div className='row group'>
          <button className='button secondary' onClick={async ()=>{ try{ await axios.post(API+'/pause'); setToasts(t=>[...t,'Paused']) } catch(e:any){ setToasts(t=>[...t,'Pause failed: '+(e?.message||e)]) } }}>Pause</button>
          <button className='button secondary' onClick={async ()=>{ try{ await axios.post(API+'/resume'); setToasts(t=>[...t,'Resumed']) } catch(e:any){ setToasts(t=>[...t,'Resume failed: '+(e?.message||e)]) } }}>Resume</button>
        </div>
        <div className='row group'>
          <button className='button secondary' onClick={async ()=>{
            const n = Number(prompt('Add how many agents?', '50')||'0');
            if (!n || n<=0) return;
            try{ const r = await axios.post(API+'/agents/add', { n }); setToasts(t=>[...t, `Added ${r.data?.added||n} agents (total=${r.data?.agents})`]) }catch(e:any){ setToasts(t=>[...t,'Add agents failed: '+(e?.message||e)]) }
          }}>Add Agents</button>
          <button className='button secondary' onClick={async ()=>{
            const c = Number(prompt('Resize to how many agents?', String(numAgents))||String(numAgents));
            if (isNaN(c) || c<0) return;
            try{ const r = await axios.post(API+'/agents/resize', { count: c }); setNumAgents(c); setToasts(t=>[...t, `Agents resized (total=${r.data?.agents})`]) }catch(e:any){ setToasts(t=>[...t,'Resize agents failed: '+(e?.message||e)]) }
          }}>Resize Agents</button>
        </div>
        <div className='row group'>
          <button className='button secondary' onClick={()=>openCorpus(1)}>Corpus</button>
          <button className='button secondary' onClick={async ()=>{ try{const r=await axios.post(API+'/jobs/start',{query}); setJobId(r.data.job_id); setJobPct(0); setToasts(t=>[...t, 'Job '+r.data.job_id+' started'])}catch(e:any){setToasts(t=>[...t, 'Job start failed: '+(e?.message||e)])}}}>Shard Run</button>
        </div>
        <div className='row group'>
          <button className='button' onClick={doSearch}>Search</button>
          <button className='button secondary' onClick={doAnswer}>Answer</button>
        </div>
      </aside>
      <main className='main'>
        <div className='card'>
          <div className='section-title'>Live pheromone network</div>
          <Plot
            data={fig.data as any}
            layout={fig.layout as any}
            style={{ width:'100%', height:'600px' }}
            onClick={(e:any)=>{
              try{
                // get point index and pick meta
                const p = e?.points?.[0];
                if (!p) return;
                if (p.data?.name !== 'docs') return; // only respond to doc points
                const idx = p.pointIndex;
                const m = Array.isArray(p.data?.customdata) ? p.data.customdata[idx] : undefined;
                if (m){ setSelectedDocs(prev=>[{...m}, ...prev].slice(0,20)); }
              }catch{}
            }}
          />
        </div>
        <div className='card'>
          <div className='section-title'>Live log</div>
          <div className='log'>{logs.join('\n')}</div>
        </div>
        {jobId && (
          <div className='card'>
            <div className='section-title'>Shard progress (job {jobId})</div>
            <div className='progress'><div className='progress-bar' style={{ width: `${jobPct}%` }} /></div>
          </div>
        )}
        <div className='card'>
          <div className='section-title'>Results</div>
          <div className='results'>
            {selectedDocs.map((m, idx) => (
              <div key={'sel'+idx} className='result-item'>
                <div style={{display:'flex',gap:'8px',alignItems:'baseline'}}>
                  <span className='score'>{Number(m?.score||0).toFixed(3)}</span>
                  <span>doc #{m?.id}</span>
                  <button className='button secondary' style={{marginLeft:'auto'}} onClick={async()=>{
                    if (m?.id == null) return
                    try { setLoadingDoc(m.id); const r = await axios.get(API+`/doc/${m.id}`); const d = r.data?.doc; setSelectedDocs(prev=> prev.map((x:any)=> x.id===m.id ? {...x, full:d?.content||'', embedding:d?.embedding||[]} : x)) } catch(e:any){ setToasts(t=>[...t,'Fetch doc failed: '+(e?.message||e)]) } finally { setLoadingDoc(null) }
                  }}>Load content & embedding</button>
                </div>
                <span className='snippet'>{String(m?.snippet||'')}</span>
                {loadingDoc===m?.id && (<div style={{opacity:0.7, fontStyle:'italic'}}>loading…</div>)}
                {m?.full && (
                  <pre style={{whiteSpace:'pre-wrap', background:'rgba(0,0,0,0.04)', padding:'8px', borderRadius:4, marginTop:6, maxHeight:200, overflow:'auto'}}>{String(m.full)}</pre>
                )}
                {Array.isArray(m?.embedding) && m.embedding.length>0 && (
                  <div style={{marginTop:6}}>
                    <div style={{fontWeight:600, opacity:0.85}}>Embedding ({m.embedding.length} dims)</div>
                    <div style={{fontFamily:'monospace', fontSize:12, maxHeight:120, overflow:'auto'}}>{String(m.embedding.slice(0,64).map((v:number)=>Number(v).toFixed(3))).replace(/,/g, ', ') }{m.embedding.length>64?' …':''}</div>
                  </div>
                )}
              </div>
            ))}
            {results.map((it, idx) => (
              <div key={idx} className='result-item'>
                <span className='score'>{Number(it.relevance_score||0).toFixed(3)}</span>
                <span>{(it.metadata?.file_path) || 'chunk'}</span>
                <span className='snippet'>{String((it.content||'')).slice(0,180)}</span>
              </div>
            ))}
          </div>
        </div>
        <div className='card'>
          <div className='section-title'>Step Report</div>
          {reports.length === 0 ? (
            <div className='results' style={{padding:10}}>No report yet.</div>
          ) : (
            <div className='results'>
              <div style={{display:'flex',gap:10,alignItems:'center', padding:'8px 12px'}}>
                <label style={{display:'flex',gap:6,alignItems:'center'}}>
                  <input type='checkbox' checked={autoScrollReport} onChange={e=>setAutoScrollReport(e.target.checked)} /> Auto-scroll latest
                </label>
              </div>
              {(() => {
                const latest = reports[reports.length-1]
                const items = Array.isArray(latest?.data?.items) ? latest.data.items : []
                return (
                  <div>
                    <div style={{padding:'8px 12px', fontWeight:700}}>Step {latest.step} • Items {items.length}</div>
                    {items.slice(0,20).map((it:any, idx:number) => (
                      <div key={'rep'+idx} className='result-item'>
                        <div style={{display:'flex',gap:8,alignItems:'baseline'}}>
                          <span className='score'>{Number(it?.embedding_score||0).toFixed(3)}</span>
                          <span style={{opacity:0.85}}>{String(it?.file_path||'file')}</span>
                          <span style={{opacity:0.6}}>lines {Array.isArray(it?.line_range)? it.line_range.join('-') : ''}</span>
                        </div>
                        <div className='snippet'>{String(it?.code_purpose||'')}</div>
                        {it?.relevance_to_query && (<div className='snippet'>why: {String(it.relevance_to_query)}</div>)}
                      </div>
                    ))}
                    <div style={{padding:10}}>
                      <button className='button secondary' onClick={()=>{
                        try{
                          const blob = new Blob([JSON.stringify(latest.data, null, 2)], {type:'application/json'})
                          const a = document.createElement('a')
                          a.href = URL.createObjectURL(blob)
                          a.download = `report_step_${latest.step}.json`
                          a.click()
                        }catch(e){/* noop */}
                      }}>Save latest JSON</button>
                    </div>
                  </div>
                )
              })()}
            </div>
          )}
        </div>
      </main>
      {showCorpus && (
        <div className='modal-backdrop' onClick={()=>setShowCorpus(false)}>
          <div className='modal' onClick={e=>e.stopPropagation()}>
            <div className='modal-header'>
              <div>Corpus Explorer (page {corpusPage}, total {corpusTotal})</div>
              <button className='button secondary' onClick={()=>setShowCorpus(false)}>Close</button>
            </div>
            <div className='modal-body'>
              <div className='filelist'>
                {corpusFiles.map((f,i)=>(<div key={i} className='file-item'>{f}</div>))}
              </div>
            </div>
          </div>
        </div>
      )}
      <div className='toasts'>
        {toasts.slice(-4).map((m,i)=>(<div key={i} className='toast'>{m}</div>))}
      </div>
    </div>
  )
}


