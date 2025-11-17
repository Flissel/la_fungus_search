// ==========================================
// Simulation Context
// Provides global state to all components
// ==========================================

import React, { createContext, useContext, useState } from 'react'
import type { Collection } from '../types'

interface ToastContextValue {
  toasts: string[]
  addToast: (message: string) => void
}

interface CollectionContextValue {
  collections: Collection[]
  setCollections: React.Dispatch<React.SetStateAction<Collection[]>>
  activeCollection: string
  setActiveCollection: React.Dispatch<React.SetStateAction<string>>
}

interface ModalContextValue {
  showCorpus: boolean
  setShowCorpus: React.Dispatch<React.SetStateAction<boolean>>
  showPrompts: boolean
  setShowPrompts: React.Dispatch<React.SetStateAction<boolean>>
  showCollections: boolean
  setShowCollections: React.Dispatch<React.SetStateAction<boolean>>
  corpusFiles: string[]
  setCorpusFiles: React.Dispatch<React.SetStateAction<string[]>>
  corpusPage: number
  setCorpusPage: React.Dispatch<React.SetStateAction<number>>
  corpusTotal: number
  setCorpusTotal: React.Dispatch<React.SetStateAction<number>>
  promptModes: string[]
  setPromptModes: React.Dispatch<React.SetStateAction<string[]>>
  promptDefaults: Record<string, string>
  setPromptDefaults: React.Dispatch<React.SetStateAction<Record<string, string>>>
  promptOverrides: Record<string, string>
  setPromptOverrides: React.Dispatch<React.SetStateAction<Record<string, string>>>
  promptModeSel: string
  setPromptModeSel: React.Dispatch<React.SetStateAction<string>>
  autoScrollReport: boolean
  setAutoScrollReport: React.Dispatch<React.SetStateAction<boolean>>
}

const ToastContext = createContext<ToastContextValue | null>(null)
const CollectionContext = createContext<CollectionContextValue | null>(null)
const ModalContext = createContext<ModalContextValue | null>(null)

export function SimulationProvider({ children }: { children: React.ReactNode }) {
  // Toast state
  const [toasts, setToasts] = useState<string[]>([])

  const addToast = (message: string) => {
    setToasts((prev) => [...prev.slice(-3), message])
  }

  // Collection state
  const [collections, setCollections] = useState<Collection[]>([])
  const [activeCollection, setActiveCollection] = useState<string>('codebase')

  // Modal state
  const [showCorpus, setShowCorpus] = useState(false)
  const [showPrompts, setShowPrompts] = useState(false)
  const [showCollections, setShowCollections] = useState(false)
  const [corpusFiles, setCorpusFiles] = useState<string[]>([])
  const [corpusPage, setCorpusPage] = useState(1)
  const [corpusTotal, setCorpusTotal] = useState(0)
  const [promptModes, setPromptModes] = useState<string[]>([])
  const [promptDefaults, setPromptDefaults] = useState<Record<string, string>>({})
  const [promptOverrides, setPromptOverrides] = useState<Record<string, string>>({})
  const [promptModeSel, setPromptModeSel] = useState<string>('deep')
  const [autoScrollReport, setAutoScrollReport] = useState<boolean>(true)

  return (
    <ToastContext.Provider value={{ toasts, addToast }}>
      <CollectionContext.Provider
        value={{ collections, setCollections, activeCollection, setActiveCollection }}
      >
        <ModalContext.Provider
          value={{
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
            autoScrollReport,
            setAutoScrollReport,
          }}
        >
          {children}
        </ModalContext.Provider>
      </CollectionContext.Provider>
    </ToastContext.Provider>
  )
}

export function useToasts() {
  const context = useContext(ToastContext)
  if (!context) throw new Error('useToasts must be used within SimulationProvider')
  return context
}

export function useCollections() {
  const context = useContext(CollectionContext)
  if (!context) throw new Error('useCollections must be used within SimulationProvider')
  return context
}

export function useModals() {
  const context = useContext(ModalContext)
  if (!context) throw new Error('useModals must be used within SimulationProvider')
  return context
}
