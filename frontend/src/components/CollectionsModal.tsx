// ==========================================
// Collections Modal Component
// Manages and displays vector collections
// ==========================================

import React, { useState, useEffect } from 'react'
import * as api from '../services/api'
import { useToasts } from '../context/SimulationContext'
import type { Collection } from '../types'

interface CollectionsModalProps {
  show: boolean
  onClose: () => void
  activeCollection: string
}

export function CollectionsModal({ show, onClose, activeCollection }: CollectionsModalProps) {
  const { addToast } = useToasts()
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (show) {
      loadCollections()
    }
  }, [show])

  const loadCollections = async () => {
    try {
      setLoading(true)
      const res = await api.fetchCollections()
      setCollections(res.collections || [])
    } catch (e: any) {
      addToast('Failed to load collections: ' + (e?.message || e))
    } finally {
      setLoading(false)
    }
  }

  const handleSwitch = async (name: string) => {
    try {
      await api.switchCollection(name)
      addToast(`Switched to collection: ${name}`)
      loadCollections()
    } catch (e: any) {
      addToast('Switch failed: ' + (e?.message || e))
    }
  }

  const handleDelete = async (name: string) => {
    if (name === activeCollection) {
      addToast('Cannot delete active collection. Switch to another collection first.')
      return
    }
    if (!confirm(`Delete collection "${name}"? This cannot be undone.`)) {
      return
    }
    try {
      await api.deleteCollection(name)
      addToast(`Deleted collection: ${name}`)
      loadCollections()
    } catch (e: any) {
      addToast('Delete failed: ' + (e?.message || e))
    }
  }

  if (!show) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>Collections Manager</div>
          <button className="button secondary" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="modal-body">
          {loading ? (
            <div>Loading collections...</div>
          ) : (
            <div className="collections-list">
              {collections.length === 0 ? (
                <div>No collections found</div>
              ) : (
                collections.map((c) => (
                  <div key={c.name} className="collection-item" style={{ padding: '12px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: c.is_active ? 700 : 400, color: c.is_active ? 'var(--primary)' : 'var(--text)' }}>
                        {c.name} {c.is_active && '(Active)'}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '4px' }}>
                        {c.point_count} points • {c.dimension}d vectors
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      {!c.is_active && (
                        <button className="button secondary" onClick={() => handleSwitch(c.name)} style={{ fontSize: '12px', padding: '6px 10px' }}>
                          Switch
                        </button>
                      )}
                      {!c.is_active && (
                        <button className="button secondary" onClick={() => handleDelete(c.name)} style={{ fontSize: '12px', padding: '6px 10px' }}>
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
