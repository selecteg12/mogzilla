export type RatingTier = { label: string; low: number; high: number }
export const RATING_TIERS: RatingTier[]
export function ratingTier(value: number): RatingTier | null
