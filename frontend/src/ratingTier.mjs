// Categories apply to the same rounded number that the interface displays.
export const RATING_TIERS = [
  { label: 'sub 3', low: 0, high: 2 },
  { label: 'sub 5', low: 2, high: 4 },
  { label: 'mtn', low: 4, high: 5 },
  { label: 'lntn', low: 5, high: 5.5 },
  { label: 'ntn', low: 5.5, high: 6 },
  { label: 'hltl', low: 6, high: 6.5 },
  { label: 'htn', low: 6.5, high: 7.5 },
  { label: 'chadlite', low: 7.5, high: 8.5 },
  { label: 'chad', low: 8.5, high: 9.5 },
  { label: 'true adam', low: 9.5, high: 10 },
]

export function ratingTier(value) {
  if (!Number.isFinite(value) || value < 0 || value > 10) return null
  const displayed = Number(value.toFixed(1))
  return RATING_TIERS.find(tier => displayed >= tier.low && (displayed < tier.high || displayed === 10 && tier.high === 10)) ?? null
}
