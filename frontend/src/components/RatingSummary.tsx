import type { RatingPart } from '../types'
import type { RatingState } from '../useGeometryRating'
import ParameterHelp from './ParameterHelp'

export default function RatingSummary({ state, selectedId, onSelect, onHover }: { state: RatingState; selectedId: string; onSelect: (id: string) => void; onHover: (id: string | null) => void }) {
  const { rating, error, busy } = state
  return <div className="sidebar-rating">
    <span className="card-kicker">РЕЙТИНГ ПРИВЛЕКАТЕЛЬНОСТИ ПО ГЕОМЕТРИИ</span>
    <div className="sidebar-score" aria-live="polite"><strong>{rating?.value != null ? rating.value.toFixed(1) : '—'}</strong><span>/ 10</span></div>
    <span className="review-badge">Автоматический расчёт · субъективная шкала</span>
    <p>Субъективная шкала выбранных пропорций. Наведите на параметр — его линии появятся на фото. Нажмите, чтобы закрепить.</p>
    {error && <p role="alert" className="error-text">Не удалось рассчитать балл: {error}</p>}
    {!rating && !error && <p role="status">{busy ? 'Рассчитываем баллы…' : 'Ожидаем измерения лица…'}</p>}
    <div className="rating-parameter-list">{rating?.parts?.map((part: RatingPart) => <div key={part.id} className={`rating-parameter ${part.id === selectedId ? 'active' : ''}`} onMouseEnter={() => onHover(part.id)} onMouseLeave={() => onHover(null)}>
      <button className="rating-parameter-button" aria-pressed={part.id === selectedId} onFocus={() => onHover(part.id)} onBlur={() => onHover(null)} onClick={() => onSelect(part.id)} title={`${part.description} ${part.summary}`}>
        <span className={`parameter-dot ${part.deviation.direction === 'within' ? 'within' : 'outside'}`} /><span className="parameter-name">{part.title}<small>{part.summary}</small></span><b>{part.score.toFixed(1)}<small>/10</small></b>
      </button><ParameterHelp title={part.title}>{`${part.description} На фото: ${part.observed.toFixed(2)}${part.unit === '°' ? '°' : ''}. Выбранный диапазон: ${part.low}–${part.high}. ${part.summary} Снижает итог на ${part.lost_contribution.toFixed(2)} балла из 10.`}</ParameterHelp>
    </div>)}</div>
    {rating?.parts && <p className="fine-print">Цвет: янтарный — вне выбранного диапазона, зелёный — внутри. Отклонение от предпочтения не доказывает наличие дефекта.</p>}
    <a className="text-button" href="#rating">Все вклады, диапазоны и веса ↓</a>
  </div>
}
