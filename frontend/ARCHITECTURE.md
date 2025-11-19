# Frontend Architecture

## Component Hierarchy

```
App (230 lines)
├── SimulationProvider (Context)
│   ├── ToastContext
│   ├── CollectionContext
│   └── ModalContext
│
└── AppContent
    ├── Sidebar
    │   ├── SettingsPanel (737 lines)
    │   │   └── Theme, Corpus Status, Simulation, Models, Chunking, Parameters, Collections
    │   ├── ControlPanel (156 lines)
    │   │   └── Start, Stop, Reset, Pause, Resume, Agents, Corpus, Jobs
    │   └── SearchPanel (47 lines)
    │       └── Search, Answer
    │
    ├── Main
    │   ├── VisualizationPanel (189 lines)
    │   │   └── 3D Pheromone Network (Plotly)
    │   ├── LogPanel (68 lines)
    │   │   └── Live Logs + Events Table
    │   ├── MetricsPanel (42 lines)
    │   │   └── Real-time Metrics Chart (Plotly)
    │   └── ResultsPanel (182 lines)
    │       └── Selected Docs, Search Results, Reports, Job Progress
    │
    ├── Modals
    │   ├── CorpusModal (42 lines)
    │   └── PromptsModal (107 lines)
    │
    └── StatusBar (48 lines)
        └── Toasts + Modal Indicator
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         App.tsx                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           SimulationProvider (Context)                │   │
│  │  - Toast notifications                                │   │
│  │  - Collection management                              │   │
│  │  - Modal state                                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  useSettings │  │ useSimulation│  │ useWebSocket │      │
│  │     Hook     │  │  State Hook  │  │     Hook     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │               │
│         └─────────────────┼──────────────────┘               │
│                           │                                  │
│         ┌─────────────────┴─────────────────┐               │
│         │                                   │               │
│    ┌────▼────┐                         ┌───▼────┐          │
│    │ UI      │                         │ Backend│          │
│    │ Components                        │ API    │          │
│    └─────────┘                         └────────┘          │
└─────────────────────────────────────────────────────────────┘

Backend Communication:
┌──────────┐     ┌──────────┐     ┌─────────┐
│ WebSocket├────►│ useWS    ├────►│Simulation│
│   /ws    │     │  Hook    │     │  State   │
└──────────┘     └──────────┘     └─────────┘

┌──────────┐     ┌──────────┐     ┌─────────┐
│   HTTP   ├────►│ api.ts   ├────►│Components│
│ Endpoints│     │ Service  │     │          │
└──────────┘     └──────────┘     └─────────┘
```

## Hook Responsibilities

### useSettings Hook (335 lines)
**Purpose**: Manages all application settings state

**State**:
- Visualization: dims, minTrail, maxEdges, redrawEvery
- Simulation: numAgents, maxIterations, exploration, pheromone
- Query: query, mode, topK, reports, multi-query
- LLM: 4 providers (Ollama, OpenAI, Google, Grok)
- Corpus: useRepo, rootFolder, maxFiles, excludeDirs, windows
- Theme: light/dark/system with localStorage

**Methods**:
- `buildSettings()`: Constructs API payload
- `updateRootFolder()`: Validates and updates folder path
- `updateTheme()`: Updates theme and persists to localStorage

### useSimulationState Hook (155 lines)
**Purpose**: Manages simulation runtime state

**State**:
- Snapshot, logs, results, reports, events
- Selected documents, corpus status
- Job progress (ID, percentage)
- Metrics: steps, avg, max, trails, results

**Methods**:
- `addLog()`, `addEvent()`, `addReport()`, `addResults()`
- `addSeedQueries()`, `updateMetrics()`
- `resetState()`: Clears all simulation data

### useWebSocket Hook (123 lines)
**Purpose**: Manages WebSocket connection and message handling

**Features**:
- Auto-connect with retry logic (1s delay)
- Handles 7 message types: snapshot, report, results, seed_queries, log, metrics, job_progress
- `sendConfig()`: Sends configuration updates

**Pattern**:
```typescript
useWebSocket({
  onSnapshot: (data) => { /* handle */ },
  onReport: (step, data) => { /* handle */ },
  onResults: (step, data, type) => { /* handle */ },
  // ... more handlers
})
```

## Service Layer

### api.ts (149 lines)
**Purpose**: Centralized HTTP API client

**Endpoints**:
- Settings: `fetchSettings()`, `updateSettings()`
- Simulation: `start`, `stop`, `pause`, `resume`, `reset`
- Agents: `addAgents()`, `resizeAgents()`
- Search: `search()`, `answer()`
- Corpus: `fetchCorpusList()`, `fetchDocument()`
- Collections: `fetchCollections()`, `switchCollection()`
- Reports: `mergeReports()`
- Jobs: `startJob()`
- Prompts: `fetchPrompts()`, `savePrompts()`

## Type System

### types/index.ts (143 lines)
**Exports**:
- `ThemeMode`: 'light' | 'dark' | 'system'
- `Snapshot`: Documents, agents, edges for visualization
- `Collection`: Name, point count, dimension, active status
- `CorpusStatus`: Docs, files, agents counts
- `SearchResult`, `ReportItem`, `Report`, `EventLog`
- `Settings`: All 50+ configuration options
- `WebSocketMessage`: Message format
- `DocMetadata`: Document with embedding and content

## Component Details

### Main Container Components

**SettingsPanel** (737 lines - largest component)
- Complete configuration UI
- Handles 4 LLM providers with conditional rendering
- Form controls for all settings
- Collection management
- Summary building

**ControlPanel** (156 lines)
- Primary simulation controls
- Agent management
- Corpus and job operations

**VisualizationPanel** (189 lines)
- 3D/2D Plotly visualization
- Theme-aware rendering
- Document click interactions
- Memoized for performance

**ResultsPanel** (182 lines)
- Multi-section results display
- Document loading with embeddings
- Report visualization
- Job progress tracking

### Utility Components

**MetricsPanel** (42 lines)
- Real-time metrics charts
- Dual Y-axis (relevance + counts)

**LogPanel** (68 lines)
- Live log streaming
- Events table with filtering

**SearchPanel** (47 lines)
- Search/Answer execution
- Minimal, focused UI

**StatusBar** (48 lines)
- Toast notifications
- Modal status indicator

### Modal Components

**CorpusModal** (42 lines)
- File list display
- Pagination info

**PromptsModal** (107 lines)
- Prompt editing interface
- Override management
- Default prompt display

## State Management Strategy

### Context API (SimulationContext.tsx)
**Global State**:
- Toasts: Notification messages
- Collections: Available collections list
- Modals: Corpus/Prompts visibility and data

**Why Context?**
- Shared across many components
- Avoids prop drilling
- Centralized management

### Local Hooks
**Component-Specific State**:
- Settings: Configuration values
- Simulation: Runtime data
- WebSocket: Connection management

**Why Hooks?**
- Encapsulated logic
- Reusable across components
- Clear separation of concerns

## File Size Analysis

| Size Range | Count | Files |
|------------|-------|-------|
| 0-50 lines | 4 | SearchPanel, StatusBar, CorpusModal, MetricsPanel |
| 51-100 lines | 1 | LogPanel |
| 101-150 lines | 3 | PromptsModal, SimulationContext, useWebSocket |
| 151-200 lines | 4 | ControlPanel, useSimulationState, ResultsPanel, VisualizationPanel |
| 201-400 lines | 2 | App (230), useSettings (335) |
| 401+ lines | 1 | SettingsPanel (737) |

## Benefits of New Architecture

### 1. Maintainability
- Find code quickly: "Where's the WebSocket logic?" → `hooks/useWebSocket.ts`
- Change features easily: "Update LLM settings" → `components/SettingsPanel.tsx`
- Test components: Each component can be tested in isolation

### 2. Reusability
- `useSettings` hook can be used in any component
- API functions shared across components
- Type definitions ensure consistency

### 3. Scalability
- Add new component: Just create in `components/`
- Add new API endpoint: Just add to `services/api.ts`
- Add new state: Extend appropriate hook

### 4. Developer Experience
- Clear file structure
- TypeScript autocomplete
- Smaller files easier to understand
- Separation of concerns

### 5. Performance
- Memoized visualizations
- Efficient context usage
- Minimal re-renders

## Migration Notes

### Breaking Changes
**None** - All functionality preserved

### New Dependencies
**None** - Uses existing packages

### Backward Compatibility
- Original App.tsx backed up as `App.tsx.backup`
- All props, state, and behavior identical
- Same UI/UX maintained

## Future Enhancements

### Potential Improvements
1. **Tests**: Add unit tests for hooks and components
2. **Lazy Loading**: Code-split large components
3. **Error Boundaries**: Add error handling components
4. **Storybook**: Component documentation
5. **Performance**: React.memo for expensive components
6. **Accessibility**: ARIA labels and keyboard navigation
7. **Mobile**: Responsive design improvements

### Recommended Next Steps
1. Test all functionality thoroughly
2. Add PropTypes or enhance TypeScript types
3. Implement error boundaries
4. Add loading states
5. Improve form validation
