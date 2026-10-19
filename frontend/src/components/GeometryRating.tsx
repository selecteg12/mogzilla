import { ChevronRight, Sparkles } from 'lucide-react'
import type { RatingState } from '../useGeometryRating'
import ParameterHelp from './ParameterHelp'

export default function GeometryRating({ state, onSelect, onHover }: { state: RatingState; onSelect: (id: string) => void; onHover: (id: string | null) => void }) {
  const { settings, rating, error, busy } = state
  return <section className="rating-section section-anchor" id="rating">
    <div className="section-heading"><div><span className="eyebrow">СУБЪЕКТИВНАЯ ШКАЛА · 0–10</span><h2>Из чего складывается рейтинг.</h2></div><Sparkles size={23} className="section-icon" /></div>
    <div className="rating-layout"><div className="rating-score-card">
      <span className="card-kicker">ПРИВЛЕКАТЕЛЬНОСТЬ ПО ВЫБРАННЫМ ПРОПОРЦИЯМ</span>
      <div className="rating-number" aria-live="polite">{rating?.value != null ? rating.value.toFixed(1) : '—'}<span>/ 10</span></div>
      <p>{busy && !rating ? 'Рассчитываем вклад каждого показателя…' : rating?.message ?? `Учтено критериев: ${rating?.used_criteria ?? '—'}. Балл отражает соответствие выбранным предпочтениям.`}</p>
      <p className="provisional-note">Автоматический результат зависит от ракурса, мимики и погрешности найденных точек.</p>
      {rating?.sensitivity && <div className="rating-sensitivity"><strong>{rating.sensitivity.low.toFixed(1)}–{rating.sensitivity.high.toFixed(1)} / 10</strong><span>Чувствительность к ошибке ориентиров</span><p>{rating.sensitivity.interpretation}</p></div>}
      <small>Это не объективная оценка человека и не модель, обученная на мнениях о внешности.</small>
    </div><div className="rating-breakdown"><h3>Где теряются баллы</h3><p>Наведите для пояснения. Нажмите на параметр, чтобы увидеть его точки и отрезки на фотографии.</p>
      {rating?.parts && <div className="rating-table-wrap"><table className="rating-table"><thead><tr><th>Критерий</th><th>Измерено</th><th>Диапазон</th><th>Балл / 10</th><th>Потеря итога</th></tr></thead><tbody>{rating.parts.map(part => <tr key={part.id} onMouseEnter={() => onHover(part.id)} onMouseLeave={() => onHover(null)}><td><div className="table-parameter"><button className="text-button" title={part.description} onClick={() => onSelect(part.id)}>{part.title}<ChevronRight size={12} /></button><ParameterHelp title={part.title}>{`${part.description} ${part.summary}`}</ParameterHelp></div></td><td>{part.observed.toFixed(2)}</td><td>{part.low}–{part.high}</td><td>{part.score.toFixed(1)}</td><td>−{part.lost_contribution.toFixed(2)}</td></tr>)}</tbody></table></div>}
      {rating?.unavailable?.map(item => <p key={item.id} className="fine-print">{settings?.criteria.find(c => c.id === item.id)?.title}: {item.reason}</p>)}
      <p className="fine-print">Потеря итога = (10 − балл критерия) × его вес / сумму весов. Она показывает, почему итог ниже 10. Эти предпочтения не являются универсальной нормой.</p>
      <p className="fine-print">{settings?.coverage}</p>
    </div></div>
    {error && <div role="alert" className="error-banner">Не удалось рассчитать рейтинг: {error}</div>}
    {settings && <div className="rating-method"><p>Внутри заданного диапазона критерий получает 10. За его границами балл линейно уменьшается до нуля. Итог — среднее с фиксированными весами. Ниже приведена формула, по которой уже выполнен расчёт.</p><code>{settings.formula}</code><p>{settings.interpretation.replace(' и редактируются', '')} <a href={settings.source} target="_blank" rel="noreferrer">Источник диапазонов ↗</a></p></div>}
  </section>
}
