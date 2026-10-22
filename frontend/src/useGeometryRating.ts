import { useEffect, useState } from 'react'
import { fetchJson } from './api'
import type { Analysis, RatingResult, RatingSettings } from './types'

// Current analysis responses already contain the score. The fallback supports
// an older API response without requiring any user action or customization.
export function useGeometryRating(analysis: Analysis | null) {
  const [fallback, setFallback] = useState<{ analysis: Analysis; settings: RatingSettings; rating: RatingResult } | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!analysis?.landmarks || !analysis.model || analysis.rating || analysis.holistic) return
    const request = new AbortController()
    setBusy(true); setError('')
    void (async () => {
      const settings = await fetchJson<RatingSettings>('/api/rating/settings', { signal: request.signal })
      const rating = await fetchJson<RatingResult>('/api/rating', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: request.signal,
        body: JSON.stringify({ dataset: analysis.model!.dataset, points: analysis.landmarks!.map(p => [p.x, p.y]), rules: settings.criteria }),
      })
      if (!request.signal.aborted) setFallback({ analysis, settings, rating })
    })().catch(e => { if (!request.signal.aborted) setError(e.message) }).finally(() => { if (!request.signal.aborted) setBusy(false) })
    return () => request.abort()
  }, [analysis])
  const current = fallback?.analysis === analysis ? fallback : null
  return { settings: analysis?.rating_settings ?? current?.settings ?? null, rating: analysis?.rating ?? current?.rating ?? null, error, busy }
}

export type RatingState = ReturnType<typeof useGeometryRating>
