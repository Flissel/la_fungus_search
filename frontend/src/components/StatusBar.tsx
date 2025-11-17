// ==========================================
// Status Bar Component
// Shows toasts and modal indicators
// ==========================================

import React from 'react'

interface StatusBarProps {
  toasts: string[]
  showCorpus: boolean
  showPrompts: boolean
}

export function StatusBar({ toasts, showCorpus, showPrompts }: StatusBarProps) {
  return (
    <>
      {/* Modal indicator */}
      {(showCorpus || showPrompts) && (
        <div
          style={{
            position: 'fixed',
            top: 10,
            right: 10,
            background: '#ef4444',
            color: 'white',
            padding: '8px 12px',
            borderRadius: 6,
            zIndex: 99999,
            fontWeight: 700,
            fontSize: 14,
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
          }}
        >
          MODAL OPEN: {showCorpus ? 'Corpus' : 'Prompts'} (Press ESC to close)
        </div>
      )}

      {/* Toasts */}
      <div className="toasts">
        {toasts.slice(-4).map((m, i) => (
          <div key={i} className="toast">
            {m}
          </div>
        ))}
      </div>
    </>
  )
}
