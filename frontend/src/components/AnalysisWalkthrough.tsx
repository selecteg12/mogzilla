import type { Analysis, Box, Metric, RatingPart } from '../types'
import type { RatingState } from '../useGeometryRating'
import FaceCanvas from './FaceCanvas'
import type { Layers } from './FaceCanvas'
import RatingSummary from './RatingSummary'
import GeometryRating from './GeometryRating'

const BASE_LAYERS: Layers = { landmarks: true, contours: false, measurements: true, original: true, symmetry: false }

export default function AnalysisWalkthrough({ photo, result, ratingState, activePart, selected, onSelect, onHover }: {
  photo: { url: string; width: number; height: number }; result: Analysis; ratingState: RatingState;
  activePart: RatingPart | null; selected: Metric | null; onSelect: (id: string, scroll?: boolean) => void; onHover: (id: string | null) => void;
}) {
  const box: Box = { x: result.box[0], y: result.box[1], width: result.box[2], height: result.box[3] }
  const inputBox: Box = { x: result.transform.origin[0], y: result.transform.origin[1], width: result.transform.side, height: result.transform.side }
  const parts = ratingState.rating?.parts ?? []
  const observedValues = activePart?.metric_ids.map(id => result.geometry?.metrics.find(metric => metric.id === id)?.value) ?? []
  const canvas = { ...photo, box, onBox: () => {}, result, disabled: true, ratingParts: parts, activePart, allCriteria: true, onCriterion: onSelect }
  return <section id="walkthrough" className="walkthrough section-anchor">
    <div className="section-heading"><div><span className="eyebrow">РАЗБОР ИМЕННО ЗАГРУЖЕННОГО СНИМКА</span><h2>От фотографии к баллу — по шагам.</h2></div></div>
    <p className="section-intro">Посмотрите, где найдено лицо, какие точки использованы и как каждый показатель повлиял на итог.</p>

    <article className="walkthrough-step" id="crop-stage">
      <header className="walkthrough-step-heading"><span>01</span><div><h3>Находим лицо на снимке</h3><p>Рамка показывает, какая часть фотографии участвовала в анализе.</p></div></header>
      <div className="walkthrough-preparation"><div className="walkthrough-region"><FaceCanvas {...canvas} box={inputBox} regionLabel="Область анализа" result={null} mode="photo" zoom={false} layers={BASE_LAYERS} selected={null} comparison={0} /><span className="stage-image-caption">Выделенный участок фотографии</span></div>
        <div className="walkthrough-copy"><div className="walkthrough-input"><img src={result.preprocessed_image} width="192" height="192" alt="Подготовленный фрагмент лица, использованный для анализа" /><div><strong>Фрагмент для анализа</strong><span>Именно по нему найдены точки лица</span></div></div>
          <p>В область должны попадать оба глаза, нос, рот и подбородок. Небольшой отступ помогает сохранить контур лица.</p><p>Если рамка захватила посторонний участок, используйте более чёткое фото анфас.</p>
        </div>
      </div>
    </article>

    <article className="walkthrough-step" id="landmarks-stage">
      <header className="walkthrough-step-heading"><span>02</span><div><h3>Определяем точки лица</h3><p>{result.landmarks?.length} точек на контуре лица, глазах, бровях, носе и губах.</p></div></header>
      <div className="walkthrough-visual-row"><div><FaceCanvas {...canvas} mode="points" zoom layers={BASE_LAYERS} selected={null} comparison={0} /></div>
        <div className="walkthrough-copy"><h4>Откуда берутся измерения</h4><p>Отмеченные точки задают границы глаз, положение зрачков, ширину носа и рта. Расстояния и углы между ними определяют значения на следующем шаге.</p><p>Наведите на точку, чтобы увидеть её положение. Сильный поворот головы, прищур и перекрытия могут сместить найденные точки.</p>
        </div>
      </div>
    </article>

    <article className="walkthrough-step section-anchor" id="geometry-stage">
      <header className="walkthrough-step-heading"><span>03</span><div><h3>Измеряем пропорции и считаем отклонения</h3><p>От найденных точек — к расстояниям, углам и сравнению с выбранными диапазонами.</p></div></header>
      <div className="walkthrough-visual-row"><div><FaceCanvas {...canvas} mode="deviations" zoom layers={BASE_LAYERS} selected={selected} comparison={0} />
        {activePart && <div className={`photo-finding ${activePart.deviation.direction === 'within' ? 'within' : 'outside'}`}><div><strong>{activePart.title}</strong><span>{activePart.score.toFixed(1)} / 10 · −{activePart.lost_contribution.toFixed(2)} итогового балла</span></div><p>На снимке <b>{activePart.observed.toFixed(2)}{activePart.unit === '°' ? '°' : ''}</b> · выбранный диапазон <b>{activePart.low}–{activePart.high}</b>. {activePart.summary}</p><p>{activePart.description}</p><small>Сплошные линии — измеренные отрезки; пунктир — базовые линии и проекции. Отклонение от предпочтения не доказывает наличие изъяна.</small></div>}
      </div><div className="walkthrough-copy walkthrough-criteria"><RatingSummary state={ratingState} selectedId={activePart?.id ?? ''} onSelect={onSelect} onHover={onHover} /></div></div>
      {activePart && <div className="walkthrough-math"><h4>Расчёт для выбранного параметра</h4><div className="math-steps"><div><b>1 · Измерение</b><code>x = {activePart.observed.toFixed(3)}{activePart.unit === '°' ? '°' : ''}</code><p>{observedValues.length > 1 ? `Среднее для двух глаз: (${observedValues.map(value => value?.toFixed(3)).join(' + ')}) / ${observedValues.length}.` : selected?.formula}</p></div><div><b>2 · Отклонение от диапазона</b><code>Δ = max({activePart.low} − x, 0, x − {activePart.high}) = {activePart.deviation.distance.toFixed(3)}</code><p>Расстояние до ближайшей границы. Внутри заданного диапазона оно равно нулю.</p></div><div><b>3 · Балл и снятие</b><code>s = 10 × max(0, 1 − Δ / {activePart.falloff}) = {activePart.score.toFixed(2)}</code><p>Снижение итогового балла: (10 − s) × вес / сумму весов = <strong>{activePart.lost_contribution.toFixed(2)}</strong>.</p></div></div></div>}
      {!activePart && selected && <div className="walkthrough-math"><h4>{selected.title}</h4><p>{selected.description}</p><code>{selected.formula}</code><p>Это измерение не включено в выбранные критерии рейтинга. Точки: {selected.point_ids.join(', ')}.</p></div>}
    </article>

    <article className="walkthrough-step walkthrough-total"><header className="walkthrough-step-heading"><span>04</span><div><h3>Суммируем вклад и получаем итоговый рейтинг</h3><p>Каждый критерий снижает итог на свою долю с учётом фиксированного веса.</p></div></header><GeometryRating state={ratingState} onSelect={id => onSelect(id, true)} onHover={onHover} /></article>
  </section>
}
