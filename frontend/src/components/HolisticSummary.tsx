import { CircleCheck, TriangleAlert } from 'lucide-react'
import type { Analysis } from '../types'
import { ratingTier } from '../ratingTier.mjs'
import RatingMascot from './RatingMascot'

export default function HolisticSummary({ result }: { result: Analysis }) {
  const rating = result.holistic
  if (!rating) return null
  const sensitive = rating.stability.status === 'sensitive'
  return <div className="holistic-summary">
    <div className="rating-headline"><div className="rating-numbers"><span className="rating-label">Твой рейт</span>
      <span className="rating-tier">{ratingTier(rating.value)?.label}</span>
      <div className="sidebar-score" aria-live="polite"><strong>{rating.value.toFixed(1)}</strong><span>/ 10</span></div>
    </div><RatingMascot value={rating.value}/></div>
    <div className="stability-section"><div><span>При изменении кадра</span><strong>{rating.stability.low.toFixed(1)}–{rating.stability.high.toFixed(1)}</strong></div><p>{sensitive ? 'Балл заметно меняется. Попробуйте ещё один нейтральный портрет.' : 'В этом тесте небольшое изменение рамки мало меняет балл.'}</p><p className="fine-print">Технический разброс на одном фото, не диапазон достоверности.</p></div>
    <div className="quality-status">{result.quality.warnings.length ? <TriangleAlert size={15} /> : <CircleCheck size={15} />}<span>{result.quality.warnings.length ? 'Есть замечания к снимку — см. ниже' : 'Снимок прошёл техническую проверку'}</span></div>
  </div>
}
