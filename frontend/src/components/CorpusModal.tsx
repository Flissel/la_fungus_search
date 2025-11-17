// ==========================================
// Corpus Modal Component
// Displays corpus file list
// ==========================================

import React from 'react'

interface CorpusModalProps {
  show: boolean
  onClose: () => void
  corpusFiles: string[]
  corpusPage: number
  corpusTotal: number
}

export function CorpusModal({ show, onClose, corpusFiles, corpusPage, corpusTotal }: CorpusModalProps) {
  if (!show) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            Corpus Explorer (page {corpusPage}, total {corpusTotal})
          </div>
          <button className="button secondary" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="modal-body">
          <div className="filelist">
            {corpusFiles.map((f, i) => (
              <div key={i} className="file-item">
                {f}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
