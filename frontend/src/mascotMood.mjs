import { ratingTier } from './ratingTier.mjs'

export function mascotMood(value) {
  const tier = ratingTier(value)
  if (!tier) return 'calm'
  if (tier.low < 4) return 'laughing'
  if (tier.low >= 9.5) return 'shocked'
  if (tier.low >= 7.5) return 'respect'
  return 'calm'
}
